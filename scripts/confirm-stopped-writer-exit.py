#!/usr/bin/env python3
"""Offline operator recovery after actual process and provider-thread termination.

Dry run is the default. Never use lease expiry or a timeout as writer-exit proof.
The DSN is read from INTERVIEW_ENGINE_SESSIONS_DSN, never a command argument.
"""
import argparse,json,os


def recover(store,*,session_id,owner_id,generation,epoch,instance,apply=False,writers_terminated=False):
    if apply and not writers_terminated:
        raise ValueError('apply requires operator confirmation that the exact writer process and all accepted provider threads are terminated')
    with store._borrow() as conn,conn.transaction():
        row=conn.execute('SELECT generation,lease_epoch,placement_instance,control FROM engine_sessions WHERE session_id=%s AND owner_id=%s::uuid FOR UPDATE',(session_id,owner_id)).fetchone()
        if (not row or row['generation']!=generation or row['lease_epoch']!=epoch
                or row['placement_instance']!=instance or not (row['control'] or {}).get('stopped')):
            raise ValueError('stopped writer identity changed or unavailable; no recovery applied')
        done=bool(row['control'].get('stop_quiescent'))
        if apply and not done:
            conn.execute("UPDATE engine_sessions SET control=control || '{\"stop_quiescent\":true,\"stop_quiescence_source\":\"operator-confirmed-writer-exit\"}'::jsonb WHERE session_id=%s AND owner_id=%s::uuid AND generation=%s AND lease_epoch=%s AND placement_instance=%s",(session_id,owner_id,generation,epoch,instance))
        return {'mode':'apply' if apply else 'dry-run','already_quiescent':done,
                'process_status':'STOPPED' if done or apply else 'STOP_REQUESTED',
                'proof_required':'Exact writer process and every accepted provider thread terminated; expiry is not proof.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',required=True);p.add_argument('--session',required=True)
    p.add_argument('--generation',required=True,type=int);p.add_argument('--epoch',required=True,type=int)
    p.add_argument('--instance',required=True);p.add_argument('--apply',action='store_true')
    p.add_argument('--writers-terminated',action='store_true');a=p.parse_args()
    from interview_engine.pg_pool import build_pool
    from interview_engine.pg_session_store import PgSessionStore
    dsn=os.environ.get('INTERVIEW_ENGINE_SESSIONS_DSN')
    if not dsn:p.error('INTERVIEW_ENGINE_SESSIONS_DSN is required')
    # Offline operator DSN. No application route or new role grants this power.
    pool=build_pool(dsn)
    try:
        print(json.dumps(recover(PgSessionStore(pool),session_id=a.session,owner_id=a.owner,
            generation=a.generation,epoch=a.epoch,instance=a.instance,
            apply=a.apply,writers_terminated=a.writers_terminated)))
    finally:pool.close()


if __name__=='__main__':main()
