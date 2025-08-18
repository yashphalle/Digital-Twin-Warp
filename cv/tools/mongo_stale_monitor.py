import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

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


def _safe_dt(val: Any) -> Optional[datetime]:
    try:
        return val if isinstance(val, datetime) else None
    except Exception:
        return None


def _get_docs_for_gid(coll, gid: int) -> List[Dict[str, Any]]:
    try:
        return list(coll.find({"global_id": gid, "status": {"$ne": "inactive"}}))
    except Exception:
        return []


def _choose_survivor(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Prefer a doc with warp_id set; then with latest last_seen; fallback to first_seen
    with_warp = [d for d in docs if d.get('warp_id') not in (None, "", 0)]
    if with_warp:
        # If multiple, pick latest last_seen among them
        docs = with_warp
    def _key(d):
        ls = _safe_dt(d.get('last_seen')) or datetime.min
        fs = _safe_dt(d.get('first_seen')) or datetime.min
        return (ls, fs)
    docs_sorted = sorted(docs, key=_key, reverse=True)
    return docs_sorted[0]


def _merge_fields(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Compute merged times_seen/first_seen/last_seen
    ts_total = 0
    fs_vals: List[datetime] = []
    ls_vals: List[datetime] = []
    for d in docs:
        try:
            ts_total += int(d.get('times_seen', 0) or 0)
        except Exception:
            pass
        fs = _safe_dt(d.get('first_seen'))
        ls = _safe_dt(d.get('last_seen'))
        if fs: fs_vals.append(fs)
        if ls: ls_vals.append(ls)
    fs_min = min(fs_vals) if fs_vals else None
    ls_max = max(ls_vals) if ls_vals else None
    return {"times_seen_total": ts_total, "first_seen_min": fs_min, "last_seen_max": ls_max}


def dedupe_by_global_id(coll, dry_run: bool = False) -> int:
    """Ensure at most one active doc per global_id by marking others inactive and merging counts.
    Returns number of documents inactivated.
    """
    try:
        pipeline = [
            {"$match": {"status": {"$ne": "inactive"}, "global_id": {"$exists": True}}},
            {"$group": {"_id": "$global_id", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}}
        ]
        dup_gids = list(coll.aggregate(pipeline))
    except Exception as e:
        logger.error(f"dedupe: aggregation failed: {e}")
        return 0

    now = _now_pst_naive()
    inactivated = 0

    for g in dup_gids:
        gid = g.get('_id')
        if gid is None:
            continue
        docs = _get_docs_for_gid(coll, gid)
        if len(docs) <= 1:
            continue
        survivor = _choose_survivor(docs)
        others = [d for d in docs if d.get('_id') != survivor.get('_id')]
        merged = _merge_fields(docs)

        # Update survivor with merged counts/timestamps
        set_fields: Dict[str, Any] = {}
        if merged["first_seen_min"]:
            set_fields["first_seen"] = merged["first_seen_min"]
        if merged["last_seen_max"]:
            set_fields["last_seen"] = merged["last_seen_max"]
        inc_fields = {"times_seen": max(0, merged["times_seen_total"] - int(survivor.get('times_seen', 0) or 0))}

        if dry_run:
            logger.info(f"[DRY] Would keep gid={gid} doc={survivor.get('_id')} and inactivate {len(others)} duplicates")
        else:
            try:
                if set_fields or (inc_fields.get('times_seen', 0) > 0):
                    coll.update_one({"_id": survivor["_id"]}, {"$set": set_fields, "$inc": inc_fields})
            except Exception as e:
                logger.warning(f"dedupe: survivor update failed for gid={gid}: {e}")

        # Inactivate others
        for d in others:
            if dry_run:
                continue
            try:
                coll.update_one({"_id": d["_id"]}, {"$set": {"status": "inactive", "inactive_since": now}})
                inactivated += 1
            except Exception as e:
                logger.warning(f"dedupe: failed to inactivate dup {d.get('_id')} for gid={gid}: {e}")

    if dup_gids:
        logger.info(f"Dedupe: processed {len(dup_gids)} global_ids; inactivated={inactivated}")
    return inactivated


def run_once(stale_threshold_s: int = 60,
             db_name: str = DEFAULT_DB,
             coll_name: str = DEFAULT_COLL,
             uri: Optional[str] = None,
             dry_run: bool = False) -> int:
    """Mark docs inactive if last_seen is older than threshold, and dedupe by global_id.
    Returns number of modified documents (inactivated by staleness), dedupe count is logged separately.
    """
    client = get_client(uri)
    if client is None:
        return 0
    try:
        coll = client[db_name][coll_name]
        total = coll.estimated_document_count()
        logger.info(f"Access OK: total documents in {db_name}.{coll_name} = {total}")

        # Step 1: Dedupe by global_id (ensure single active doc per ID)
        try:
            _ = dedupe_by_global_id(coll, dry_run=dry_run)
        except Exception as e:
            logger.error(f"dedupe step failed: {e}")

        # Step 2: Relative cutoff based on most recent timestamp in the collection
        latest_seen = None
        basis_field = 'last_seen'
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

