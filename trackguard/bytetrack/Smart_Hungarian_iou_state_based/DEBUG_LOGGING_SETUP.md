# Debug Logging Setup

## File Log Location
`log-debug.txt` (root directory)

## Logging Configuration

### Setup
- File handler: `log-debug.txt` (mode='w', overwrite setiap run)
- Console handler: INFO level untuk important messages
- Debug level: Semua detail logging

### What is Logged

#### 1. Track State Distribution (Setiap Frame)
```
[FRAME X] Track state distribution:
  Tracked: N | Lost: M | Unconfirmed: K | Ghost: L
  Lost track IDs: [(id, misses, hits, time_since_update), ...]
  Ghost track IDs: [(id, misses, hits, confidence), ...]
```

#### 2. Stage 1 Matching
```
[FRAME X] Stage 1: N matches (tracked=K, lost_recovered=M), U unmatched
[FRAME X] Stage 1: Lost track ID matched with det Y (will re-activate)
[FRAME X] Lost track ID unmatched in Stage 1 (misses=N)
```

#### 3. Re-activation Events
```
[FRAME X] ⭐ RE-ACTIVATE: Track ID re-activated from LOST state (ID preserved, misses=N)
[FRAME X] ⭐ RE-ACTIVATE: Track ID re-activated from GHOST state (ID preserved)
[RE-ACTIVATE] Track ID re-activated (new_id=False, old_state=lost, ...)
```

#### 4. Track Creation
```
[FRAME X] ⭐ NEW TRACK CREATED: Track ID N from detection M (conf=X.XX)
```

#### 5. Track State Transitions
```
[FRAME X] Track ID state: active → GHOST (hits=N, stability=X.XX, misses=M)
[FRAME X] Track ID state: active → LOST (hits=N, stability=X.XX, misses=M)
[FRAME X] Track ID TERMINATED (misses=N >= threshold)
```

#### 6. Track Updates
```
[FRAME X] Track ID updated (state=active, hits=N)
```

## How to Use

1. Run evaluation:
```bash
python mot_evaluator.py
```

2. Check log file:
```bash
# Windows
type log-debug.txt

# Linux/Mac
cat log-debug.txt
```

3. Search for specific patterns:
- `RE-ACTIVATE` - Track recovery events
- `NEW TRACK CREATED` - Track creation (potential ID switches)
- `Lost track` - Lost track handling
- `TERMINATED` - Track termination

## Analysis Tips

### To find ID switch causes:
1. Search for "NEW TRACK CREATED" near frames where tracks should be recovered
2. Check if lost tracks are being matched (search "lost_recovered")
3. Verify re-activation is happening (search "RE-ACTIVATE")

### To understand track lifecycle:
1. Search for specific track ID: `Track 123`
2. Track state transitions: `state: active → LOST`
3. Track termination: `TERMINATED`

## Expected Patterns

### Good Pattern (Lost track recovered):
```
[FRAME 100] Track 5 state: active → LOST (misses=1)
[FRAME 101] Lost track 5 matched with det 3 (will re-activate)
[FRAME 101] ⭐ RE-ACTIVATE: Track 5 re-activated from LOST state
```

### Bad Pattern (ID switch - new track created instead):
```
[FRAME 100] Track 5 state: active → LOST (misses=1)
[FRAME 101] Lost track 5 unmatched in Stage 1 (misses=2)
[FRAME 101] ⭐ NEW TRACK CREATED: Track 6 from detection 3
```
This indicates lost track was not recovered, causing ID switch.

