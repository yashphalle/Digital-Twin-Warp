# 🔧 BoT-SORT Tracking Parameter Tuning Guide

## 🎯 **Key Parameters to Fix Tracking Issues**

### 1. **Detection Thresholds** (Most Critical)

```python
# Current values in WAREHOUSE_BOTSORT_CONFIG:
'track_high_thresh': 0.3,      # Lower = easier to continue existing tracks
'track_low_thresh': 0.05,      # Lower = better recovery from occlusion  
'new_track_thresh': 0.6,       # Higher = harder to create new tracks
```

**If objects keep losing tracking:**
- **Decrease** `track_high_thresh` to 0.2 or 0.25
- **Decrease** `track_low_thresh` to 0.03 or 0.05

**If too many false tracks are created:**
- **Increase** `new_track_thresh` to 0.7 or 0.8

### 2. **Persistence Parameters**

```python
'track_buffer': 60,            # Frames to keep lost tracks
'max_age': 90,                # Max frames before deletion
```

**If objects disappear too quickly:**
- **Increase** `track_buffer` to 90-120
- **Increase** `max_age` to 120-150

**If tracking is too slow/laggy:**
- **Decrease** `track_buffer` to 30-45
- **Decrease** `max_age` to 60-75

### 3. **Matching Parameters**

```python
'match_thresh': 0.6,          # IoU threshold for matching
'proximity_thresh': 0.4,      # Spatial constraint for ReID
'appearance_thresh': 0.15,    # Appearance similarity threshold
```

**If objects switch IDs frequently:**
- **Decrease** `match_thresh` to 0.5 or 0.4
- **Decrease** `appearance_thresh` to 0.1 or 0.12

**If tracking is too loose (wrong matches):**
- **Increase** `match_thresh` to 0.7 or 0.8
- **Increase** `appearance_thresh` to 0.2 or 0.25

### 4. **Stability Parameters**

```python
'min_hits': 1,                # Frames needed to confirm track
'fuse_score': True,           # Combine confidence + IoU
'gmc_method': 'sparseOptFlow' # Camera motion compensation
```

**If too many short-lived tracks:**
- **Increase** `min_hits` to 2 or 3

**If tracking is too conservative:**
- **Decrease** `min_hits` to 1

## 🔄 **Cross-Camera Tracking Parameters**

### Smart Persistence Cross-Camera Settings:

```python
# In SmartPersistenceManager
'cross_camera_penalty': 0.9,     # Penalty for cross-camera matching
'similarity_threshold': 0.5,     # Minimum similarity for cross-camera match
'time_decay_seconds': 30,        # Time window for cross-camera matching
```

**Cross-Camera Similarity Weights:**
```python
size_ratio * 0.4 +           # Size similarity (most important)
conf_similarity * 0.2 +      # Confidence consistency  
time_factor * 0.3 +          # Temporal proximity
class_match * 0.1            # Class consistency
```

## 🧪 **Testing Different Configurations**

### Quick Test Configurations:

**Configuration A: Conservative (Stable but may lose tracks)**
```python
'track_high_thresh': 0.4,
'track_low_thresh': 0.1, 
'new_track_thresh': 0.7,
'track_buffer': 90,
'max_age': 120,
'match_thresh': 0.7,
'appearance_thresh': 0.2,
'min_hits': 2
```

**Configuration B: Aggressive (May create false tracks but keeps tracking)**
```python
'track_high_thresh': 0.2,
'track_low_thresh': 0.03,
'new_track_thresh': 0.5,
'track_buffer': 60,
'max_age': 90,
'match_thresh': 0.5,
'appearance_thresh': 0.1,
'min_hits': 1
```

**Configuration C: Balanced (Recommended starting point)**
```python
'track_high_thresh': 0.3,
'track_low_thresh': 0.05,
'new_track_thresh': 0.6,
'track_buffer': 60,
'max_age': 90,
'match_thresh': 0.6,
'appearance_thresh': 0.15,
'min_hits': 1
```

## 🎯 **Troubleshooting Common Issues**

### Issue: Objects lose tracking frequently
**Solution:**
- Lower `track_high_thresh` to 0.2-0.25
- Lower `track_low_thresh` to 0.03-0.05
- Increase `track_buffer` to 90-120

### Issue: Too many false tracks created
**Solution:**
- Raise `new_track_thresh` to 0.7-0.8
- Increase `min_hits` to 2-3
- Raise `appearance_thresh` to 0.2-0.3

### Issue: Track IDs switch between objects
**Solution:**
- Lower `match_thresh` to 0.4-0.5
- Lower `appearance_thresh` to 0.1-0.12
- Enable `fuse_score: True`

### Issue: Cross-camera tracking not working
**Solution:**
- Check Redis connection
- Verify camera neighbor mapping
- Lower cross-camera similarity threshold to 0.3-0.4
- Check unmatched detection extraction

## 🚀 **How to Apply Changes**

1. **Edit the configuration** in `test_pipline_with_BoT_persistence.py`
2. **Restart the system** to apply new parameters
3. **Monitor the logs** for tracking behavior
4. **Adjust incrementally** - change one parameter at a time
5. **Test for 2-3 minutes** before making more changes

## 📊 **Monitoring Tracking Performance**

Watch for these log messages:
- `🎨 Color Debug: Track X age Y → Color` (color transitions)
- `🔄 Cross-camera match: Track X from Camera Y → Camera Z` (cross-camera)
- `👁️ TRACK DETECTED` / `⏳ TRACK BUILDING` (consecutive detection progress)
- `🆕 NEW TRACK READY FOR DB` (database insertion)

Good tracking should show:
- Consistent track IDs over time
- Smooth color transitions (Yellow → Orange at 20 frames)
- Cross-camera matches for objects moving between cameras
- Minimal track ID switching
