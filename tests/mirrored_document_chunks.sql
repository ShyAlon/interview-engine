-- Mirrored schema/functions from delivery migration20261031210000_document_chunks.sql.
-- No RLS/auth emulation here; complete migrated FORCE-RLS coverage is in delivery.
-- Append stores only new bytes. Base overwrites start a fresh chunk epoch.
ALTER TABLE public.generator_documents ADD COLUMN chunk_epoch BIGINT NOT NULL DEFAULT 0;
ALTER TABLE public.generator_documents ADD COLUMN chunk_count BIGINT NOT NULL DEFAULT 0;
ALTER TABLE public.generator_documents ADD COLUMN chunk_bytes BIGINT NOT NULL DEFAULT 0 CHECK(chunk_bytes BETWEEN 0 AND 67108864);
CREATE TABLE public.document_chunks (
 execution_session_id TEXT NOT NULL,document_key TEXT NOT NULL,epoch BIGINT NOT NULL,ordinal BIGINT NOT NULL,
 payload BYTEA NOT NULL CHECK(octet_length(payload)<=67108864),sha256 TEXT NOT NULL CHECK(sha256~'^[a-f0-9]{64}$'),
 PRIMARY KEY(execution_session_id,document_key,epoch,ordinal),
 FOREIGN KEY(execution_session_id,document_key) REFERENCES public.generator_documents(execution_session_id,document_key) ON DELETE CASCADE
);
-- Epoch reset only on a base overwrite; append updates metadata, not payload.
CREATE FUNCTION public.requi_document_reset() RETURNS trigger
LANGUAGE plpgsql SET search_path='' AS $$
BEGIN
 IF TG_OP='UPDATE' THEN NEW.chunk_epoch=OLD.chunk_epoch+1; END IF;
 NEW.chunk_count=0;NEW.chunk_bytes=0;
 RETURN NEW;
END $$;
CREATE TRIGGER document_reset BEFORE UPDATE OF payload ON public.generator_documents FOR EACH ROW EXECUTE FUNCTION public.requi_document_reset();
-- Invoker functions retain the original tenant RLS and caller's worker/lease fence.
CREATE FUNCTION public.requi_document_append(sid TEXT,gen INT,key TEXT,raw BYTEA,owner UUID,auth_session UUID,stamp DOUBLE PRECISION) RETURNS void
LANGUAGE plpgsql SET search_path='' AS $$
DECLARE d public.generator_documents;
BEGIN
 INSERT INTO public.generator_documents(execution_session_id,owner_id,auth_session_id,generation,document_key,payload,sha256,updated_at)
 VALUES(sid,owner,auth_session,gen,key,''::bytea,encode(sha256(''::bytea),'hex'),stamp) ON CONFLICT DO NOTHING;
 SELECT * INTO d FROM public.generator_documents WHERE execution_session_id=sid AND document_key=key AND generation=gen FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'document generation unavailable'; END IF;
 IF raw IS NULL OR octet_length(COALESCE(d.payload,''::bytea))+d.chunk_bytes+octet_length(raw)>67108864 THEN RAISE EXCEPTION 'bounded document bytes required'; END IF;
 INSERT INTO public.document_chunks VALUES(sid,key,d.chunk_epoch,d.chunk_count+1,raw,encode(sha256(raw),'hex'));
 UPDATE public.generator_documents SET chunk_count=d.chunk_count+1,chunk_bytes=d.chunk_bytes+octet_length(raw),updated_at=stamp,auth_session_id=auth_session WHERE execution_session_id=sid AND document_key=key;
END $$;
CREATE FUNCTION public.requi_document_parts(sid TEXT,gen INT,key TEXT)
RETURNS TABLE(payload BYTEA,sha256 TEXT,ordinal BIGINT,total BIGINT,updated_at DOUBLE PRECISION)
LANGUAGE sql SET search_path='' AS $$
 SELECT d.payload,d.sha256,0::bigint,d.chunk_count,d.updated_at FROM public.generator_documents d
 WHERE d.execution_session_id=sid AND d.generation=gen AND d.document_key=key
 UNION ALL
 SELECT c.payload,c.sha256,c.ordinal,d.chunk_count,d.updated_at FROM public.document_chunks c JOIN public.generator_documents d
 ON d.execution_session_id=c.execution_session_id AND d.document_key=c.document_key AND d.chunk_epoch=c.epoch
 WHERE d.execution_session_id=sid AND d.generation=gen AND d.document_key=key
 ORDER BY 3
$$;
