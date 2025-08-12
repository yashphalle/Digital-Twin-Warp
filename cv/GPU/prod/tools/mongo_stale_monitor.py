import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

logger = logging.getLogger("MongoStaleMonitor")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Defaults; can be overridden by function args or RuntimeConfig
DEFAULT_DB = "warp"
DEFAULT_COLL = "detections"

# Try to import app runtime defaults
try:
    from GPU.prod.config.runtime import RuntimeConfig
    DEFAULT_DB = getattr(RuntimeConfig, 'db_database', DEFAULT_DB)
    DEFAULT_COLL = getattr(RuntimeConfig, 'db_collection', DEFAULT_COLL)
    DEFAULT_URI = getattr(RuntimeConfig, 'db_uri', None)
except Exception:
    DEFAULT_URI = None


def _now_pst_naive() -> datetime:
    now_utc = datetime.now(timezone.utc)
    pst = now_utc.astimezone(timezone(timedelta(hours=-8)))
    return pst.replace(tzinfo=None)


def get_client(uri: Optional[str] = None):
    # Accept multiple env var names for compatibility and fallback to RuntimeConfig
    uri = (
        uri
        or os.getenv("MONGODB_ONLINE_URI")
        or os.getenv("MONGO_ATLAS_URI")
        or os.getenv("MONGODB_ATLAS_URI")
        or DEFAULT_URI
    )
    if not uri:
        logger.error("No Mongo URI provided (set MONGODB_ONLINE_URI or configure RuntimeConfig.db_uri)")
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        return client
    except Exception as e:
        logger.error(f"Mongo connect failed: {e}")
        return None


def run_once(stale_threshold_s: int = 60,
             db_name: str = DEFAULT_DB,
             coll_name: str = DEFAULT_COLL,
             uri: Optional[str] = None,
             dry_run: bool = False) -> int:
    """Mark docs inactive if last_seen is older than threshold.
    Returns number of modified documents.
    """
    client = get_client(uri)
    if client is None:
        return 0
    try:
        coll = client[db_name][coll_name]
        total = coll.estimated_document_count()
        logger.info(f"Access OK: total documents in {db_name}.{coll_name} = {total}")
        # Relative cutoff based on most recent timestamp in the collection
        # Prefer 'last_seen'; fall back to 'first_seen' if needed
        latest_seen = None
        # Try last_seen
        cur = coll.find({"last_seen": {"$exists": True}}, {"last_seen": 1}).sort("last_seen", -1).limit(1)
        lst = list(cur)
        if lst and 'last_seen' in lst[0]:
            latest_seen = lst[0]['last_seen']
            basis_field = 'last_seen'
        else:
            # Fallback to first_seen
            cur2 = coll.find({"first_seen": {"$exists": True}}, {"first_seen": 1}).sort("first_seen", -1).limit(1)
            lst2 = list(cur2)
            if lst2 and 'first_seen' in lst2[0]:
                latest_seen = lst2[0]['first_seen']
                basis_field = 'first_seen'
        if latest_seen is None:
            logger.info("No documents with last_seen/first_seen found; skipping this run")
            return 0
        cutoff = latest_seen - timedelta(seconds=stale_threshold_s)
        now = _now_pst_naive()
        query = {
            "status": {"$ne": "inactive"},
            "last_seen": {"$exists": True, "$lt": cutoff}
        }
        update = {"$set": {"status": "inactive", "inactive_since": now}}
        if dry_run:
            count = coll.count_documents(query)
            logger.info(f"[DRY] Would mark inactive (relative): {count} | cutoff={cutoff} | basis={basis_field}")
            return 0
        res = coll.update_many(query, update)
        modified = getattr(res, 'modified_count', 0)
        logger.info(f"Marked inactive (relative): {modified} | cutoff={cutoff} | basis={basis_field}")
        return int(modified or 0)
    except Exception as e:
        logger.error(f"run_once failed: {e}")
        return 0
    finally:
        try:
            client.close()
        except Exception:
            pass


def main(stale_threshold_s: int = 60,
         run_interval_s: int = 60,
         db_name: str = DEFAULT_DB,
         coll_name: str = DEFAULT_COLL,
         uri: Optional[str] = None,
         dry_run: bool = False):
    logger.info(f"Starting Mongo stale monitor: threshold={stale_threshold_s}s interval={run_interval_s}s db={db_name}.{coll_name}")
    while True:
        run_once(stale_threshold_s=stale_threshold_s, db_name=db_name, coll_name=coll_name, uri=uri, dry_run=dry_run)
        time.sleep(max(5, run_interval_s))

