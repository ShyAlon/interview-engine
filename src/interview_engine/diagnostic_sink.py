"""Safe bounded review evidence degradation, no dependency on the service."""
import logging
import threading

class SinkStatus:
    def __init__(self,source):self.source=source;self._warned=False;self._lock=threading.Lock()
    def failed(self,error):
        with self._lock:
            if self._warned:return
            self._warned=True
        try:
            logging.getLogger(__name__).warning('review.evidence.degraded source=%s category=%s',self.source,type(error).__name__)
        except Exception:pass  # evidence cannot replace the authoritative result
