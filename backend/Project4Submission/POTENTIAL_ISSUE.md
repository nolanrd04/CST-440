# Potential Issue: Like & Mute Detection Failure

## Problem
After 159 test runs on float32 model:
- **call, dislike, ok** → detecting well (70-100% accuracy)
- **like, mute** → never detected correctly, always predicted as call/ok

### Test Results Summary (159 classifications)
```
PATTERNS:
  call 100% → any hand position, both hands
  dislike 70% → center, cover face, both hands
  ok 75-99% → left side, left hand
  ok 81% → left hand on side
  call 80-95% → right hand (any position)
  
FAILURES:
  like → ALWAYS predicted as call/ok, never correct
  mute → ALWAYS predicted as call, never correct
  
CORRECT PREDICTIONS:
  ✓ call (multiple hand positions)
  ✓ dislike (when covering face)
  ✓ ok (consistent with left hand use)
  ✗ like (0% accuracy - NEVER detected)
  ✗ mute (0% accuracy - NEVER detected)
```

## Root Cause
Training data for "like" and "mute" likely lacks **visual distinctiveness**:
- Only 50 samples each (likely collected under similar conditions)
- Insufficient variation in hand position, distance, lighting, angle
- Model learned call/dislike/ok features but not what makes like/mute unique

## Solution
1. **Recollect like & mute** with 300-400 samples each
   - Vary hand position (center, left, right, below face)
   - Vary distance (20cm, 30cm, 50cm)
   - Vary lighting (bright, dim, natural, artificial)
   - Keep call/dislike/ok from current dataset (they work)

2. **Combine datasets** and retrain with QAT
   - Old: call (50), dislike (50), ok (50) ✅
   - New: like (400), mute (400) ⚠️
   - Total: 950 samples

## Test First: Arduino vs Float32
Before recollecting, verify float32 also fails on Arduino:
1. Upload current `gesture_model.tflite` to Arduino
2. Test like & mute same way as webcam
3. If same result (like/mute fail) → confirms not quantization issue
4. If different result → suggests preprocessing/framing issue

## Next Steps
- [ ] Test Arduino with current model on like/mute
- [ ] Recollect like & mute (300-400 samples each)
- [ ] Combine datasets
- [ ] Retrain with QAT
- [ ] Verify improved accuracy
