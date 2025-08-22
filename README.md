# Digital Twin Warp

We built this real-time vision system for our warehouse at WARP. It tracks objects across our 11 security cameras and gives us a live view of everything moving through the facility.

## What our system does

We have 11 cameras covering our warehouse floor, and this system watches all of them simultaneously to track packages and shipments. When something moves from one camera zone to another, the system follows it and maintains a single identity.

Here's how it works: our cameras feed video → we detect objects → track them frame by frame → match them across camera zones → save everything to our database → display it on our dashboard

### Our main capabilities

- **11-camera tracking** - Monitors our entire warehouse floor simultaneously
- **Object detection** - We use YOLO models running on our GPU servers
- **Cross-camera following** - Tracks objects as they move between our camera zones
- **Dual processing modes** - We can run per-camera trackers or batch everything together
- **Background data saving** - Stores tracking data without impacting real-time performance
- **Live dashboard** - Web interface plus optional local GUI for our operators

### What we use it for

**Real-time shipment tracking** - Our operators can see exactly where any package is right now, updated live as things move.

**Quick package location** - Instead of walking around looking for shipments, we just search our system and get the exact location.

**Robot integration** - We feed this tracking data into our automated systems and robotic workflows.

**Customer visibility** - We can give our shippers detailed tracking information about their packages moving through our facility.

## How our system works

### Our processing pipeline

1. **Capture** - We grab frames from our RTSP camera streams, skipping frames when needed for performance
2. **Detection** - We run YOLO on batches of frames to find packages and objects
3. **Tracking** - We use BoT-SORT to follow objects from frame to frame
4. **Cross-camera matching** - We match objects between cameras using visual features
5. **Database** - We save all tracking data to our MongoDB in batches
6. **Display** - We show everything on our web dashboard and optional local GUI

### Our design approach

We built this system around a few key principles:

- **Camera-specific IDs** - Each of our cameras uses its own ID range (camera 8 uses 8001, 8002, etc.) so there's never any confusion about which camera detected what
- **Smart cross-camera matching** - We use visual features with a 0.5 similarity threshold, plus we mapped out which of our cameras are neighbors to each other
- **Background data saving** - Database writes happen in the background so they never slow down our real-time tracking
- **Stable database format** - We keep the same schema so our existing frontend and backend code doesn't break
- **Batched processing** - We process multiple cameras together to get better performance from our GPUs

### Our warehouse camera layout

We have 11 cameras arranged across our warehouse floor. The system knows which cameras are neighbors so it can better track objects moving between zones:

- Camera 1: neighbors [2,5] | Camera 2: neighbors [1,3,6] | Camera 3: neighbors [2,4,7] | Camera 4: neighbors [3]
- Camera 5: neighbors [1,6,8] | Camera 6: neighbors [5,7,2,9] | Camera 7: neighbors [6,3,10]
- Camera 8: neighbors [5,9] | Camera 9: neighbors [8,10,6] | Camera 10: neighbors [9,11,7] | Camera 11: neighbors [10]

### How we filter out noise

Our system filters out bad detections and noise:

- Area filtering: objects must be 15,000-1,000,000 pixels (filters out tiny specs and huge false detections)
- Size limits: we ignore anything bigger than 15 feet in any direction (probably not a package)
- Grid filtering: we use a 3x3 spatial grid to remove duplicate detections
- We store bounding boxes as 4 corners: `[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]`

### Our performance settings

- **Cross-camera matching**: We use 0.5 similarity threshold for matching objects between cameras
- **Database writes**: We only save objects after 5 consecutive detections to avoid false positives
- **Batch writes**: We save to database every 2 seconds (INSERT for new tracks, UPDATE for existing ones)
- **Frame sampling**: We process 1 out of every 20 frames to hit ~5 FPS processing from our 20 FPS camera feeds
- **Queue management**: We always use the latest frame when our queues get full

## Our codebase structure

### Backend API
```
backend/
├── live_server.py          # Our FastAPI service (runs on localhost:8000)
└── requirements.txt        # Python packages we need
```

### Computer vision pipeline
```
cv/
├── app.py                  # Main entry point - we run this to start tracking
├── buffers/
│   └── latest_frame_store.py    # Keeps the latest frame from each of our cameras
├── capture/
│   └── camera_worker.py         # Connects to our RTSP cameras
├── config/
│   ├── runtime.py              # Our main settings file
│   ├── config.py               # Database connection settings
│   ├── warehouse_botsort.yaml  # Tracker tuning parameters
│   ├── warehouse_calibration_camera_*.json  # Our camera positions (1-11)
│   ├── warehouse_config.py     # Our warehouse layout and camera neighbors
│   └── warp_main.json         # App config
├── db/
│   └── mongo_writer.py         # Saves data to our MongoDB every 2 seconds
├── detection/
│   └── detector_worker.py      # Runs YOLO detection on our camera batches
├── embedding/
│   └── embedding_service.py    # Extracts features for cross-camera matching
├── gui/
│   └── gui_display.py          # Shows our camera grid with bounding boxes
├── reid/
│   ├── association_router.py   # Coordinates cross-camera matching
│   ├── feature_worker.py       # Extracts visual features from detections
│   ├── reid_worker.py          # Matches objects between our cameras
│   ├── redis_client.py         # Our Redis connection
│   └── redis_writer.py         # Saves features to Redis
├── tools/
│   ├── mongo_stale_monitor.py  # Marks old detections as inactive
│   └── run_bench.py            # Performance testing tools
├── tracking/
│   ├── tracker_manager.py      # Per-camera tracking
│   ├── tracking_orchestrator.py # Batched tracking across all our cameras
│   ├── tracker_core.py         # Core BoT-SORT tracking logic
│   └── gpu_gmc.py             # Motion compensation
├── pipelines/                  # Our experimental code
├── modules/                    # Utility functions we wrote
├── custom_yolo.pt             # Our main object detection model
└── yolov8n-cls.pt            # Feature extraction model
```

### Web dashboard
```
frontend/
├── package.json               # npm dependencies and scripts
├── vite.config.ts            # Vite build config
├── src/                      # Our React app source
│   ├── App.tsx
│   ├── components/
│   ├── hooks/
│   └── types/
├── public/                   # Static files
├── index.html               # Main HTML file
├── Dockerfile               # Container build
└── README.md               # Frontend documentation
```

### Deployment files
```
├── Dockerfile               # Backend container
├── Dockerfile.backend       # Backend with all dependencies
└── deploy.sh               # Our Linux deployment script
```

## Running our system

### What you'll need
- Python 3.8 or newer
- GPU with CUDA (we recommend this but it'll work on CPU too)
- Redis server running
- MongoDB (we use Atlas cloud but local works too)
- Node.js 16+ if you want to run our web dashboard

### How to run it

#### Start tracking (this is what we usually run)
```bash
# Track all 11 of our cameras with everything enabled
python cv/app.py --mode tracking --all --batched --reid --db --gui

# Just track specific cameras (useful for testing)
python cv/app.py --mode tracking --cameras "8" --gui
python cv/app.py --mode tracking --cameras "1,3,5-7" --batched --reid

# Run without GUI (we use this on our servers)
python cv/app.py --mode tracking --all --batched --reid --db --no-gui
```



#### Start our web API
```bash
# Run locally for development
python backend/live_server.py

# Or use Docker for deployment
docker build -t warehouse-backend -f Dockerfile .
docker run -p 8000:8000 warehouse-backend
```

#### Start our web dashboard
```bash
cd frontend
npm install
npm run dev  # Usually opens on http://localhost:5173
```

#### Deploy everything (Linux only)
```bash
# Check the script first to make sure paths are right, then run it
./deploy.sh
```

## Our configuration

### Main settings (`cv/config/runtime.py`)

This file has most of our important settings:

#### Camera setup
- `active_cameras`: Which of our cameras to use
- `frame_skip`: We skip frames for better performance (higher = faster but less accurate)
- `resize`: We resize frames before processing to save GPU memory
- `latest_store_depth`: How many frames we keep in memory per camera

#### Detection settings
- `model_path`: Which YOLO model we use (default: `cv/custom_yolo.pt`)
- `device`: Which GPU we use (`cuda:0` for our first GPU, `cpu` for CPU-only)
- `confidence`: How confident our detector needs to be (0.0-1.0)
- `max_batch`: How many frames we process at once
- `batch_window_ms`: How long we wait to fill a batch
- `detection_batch_mode`: We use `"latest_full"` for best performance

#### Tracking settings
- `match_thresh`: How similar objects need to be to match (0.5 works well for us)
- `min_hits`: How many detections before we trust a track
- `max_age`: How long we keep tracks that disappear
- `track_buffer`: Internal buffer size
- `use_batched_tracking`: We use batched mode (recommended: true)

#### Cross-camera matching
- `reid_enabled`: Whether we turn on cross-camera matching
- `redis_uri`: Our Redis server address
- `reid_similarity_threshold`: How similar objects need to be across our cameras (0.5)
- `reid_same_cam_window_s`: Time window for same-camera matching
- `reid_neighbor_window_s`: Time window for neighbor camera matching
- `reid_topk`: How many candidates we consider

#### Database settings
- `db_enabled`: Whether we save to database
- `db_uri`: Our MongoDB connection string
- `db_database`: Our database name
- `db_collection`: Our collection name
- `db_age_threshold`: We only save tracks after this many detections
- `db_batch_interval_s`: How often we save (2.0 seconds works well for us)

### Database switching (`cv/config/config.py`)
- `USE_LOCAL_DATABASE`: We set this to true for local MongoDB, false for Atlas
- Environment variables: `MONGODB_LOCAL_URI`, `MONGODB_ONLINE_URI`, etc.

### Tracker tuning (`cv/config/warehouse_botsort.yaml`)
- `iou_strong`: Higher values (~0.80) make our tracking more strict
- `amb_topk`: Lower values (~1) reduce ambiguous matches

## Performance tuning

### What we get
- We target 5 FPS processing from our 20 FPS camera feeds
- Our system handles all 11 cameras without dropping frames
- We get real-time performance for live tracking in our warehouse

### How we make it faster

1. **Adjust frame skipping** - We increase `frame_skip` if it's running too slow
2. **Tune batch size** - Bigger `max_batch` uses more GPU memory but can be faster
3. **Disable features for testing** - We use `--no-reid` and `--no-db` to see what's slowing us down
4. **Turn off GUI** - We use `--no-gui` on our servers
5. **Try different models** - Smaller YOLO models are faster but less accurate

### How we debug performance issues
- We set `log_level: DEBUG` in config for detailed logs
- We run `cv/tools/run_bench.py` to test performance
- We check GPU usage with `nvidia-smi`
- We make sure our Redis and MongoDB are responding

## Our system components

### What each part does in our warehouse

**Camera capture**
- `CameraWorker` connects to our RTSP streams and grabs frames
- `LatestFrameStore` keeps the most recent frame from each of our cameras
- Handles frame skipping and resizing

**Object detection**
- `DetectorWorker` runs YOLO on batches of frames from our cameras
- Uses our GPU acceleration when available
- Filters out objects that are too big/small or in wrong areas of our warehouse

**Tracking**
- `TrackerManager` handles per-camera tracking
- `TrackingOrchestrator` does batched tracking across all our cameras
- Assigns persistent IDs and follows objects over time in our facility

**Cross-camera matching**
- `FeatureWorker` extracts visual features from detected objects
- `ReIDWorker` matches objects between our cameras
- `RedisClient` stores features for fast lookup
- `AssociationRouter` coordinates everything

**Database**
- `MongoWriterThread` saves our tracking data in batches
- Keeps the same database format so our existing code still works
- `mongo_stale_monitor.py` cleans up old data

**Display**
- `gui_display.py` shows our live camera feeds with bounding boxes
- Our frontend is a React web app that shows tracking data
- Both show FPS and system status

## Notes about our setup

### Development notes
- `cv/Extra` has our experimental code - we don't use it for production
- We always use `cv/app.py` as the main entry point
- We don't change the database schema or our frontend will break
- Our coordinate transformation code is already tested, we just reuse it

### Our camera setup
- Our cameras use URLs like `rtsp://admin:wearewarp!@104.181.138.5:556X/Streaming/channels/1` where X is the camera number
- We can switch to local URLs (`192.168.x.x`) if needed
- We have a config flag to choose between remote and local

### Our models
- `custom_yolo.pt` is our main detection model
- `yolov8n-cls.pt` extracts features for cross-camera matching
- Both use full precision (FP32) for better accuracy in our warehouse