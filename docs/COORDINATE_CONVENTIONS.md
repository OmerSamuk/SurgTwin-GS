# Coordinate Conventions

## Internal Camera Convention

SurgTwin-GS uses OpenCV camera coordinates:

```
+X = right
+Y = down
+Z = forward (into the scene)
```

## Pose Convention

- `c2w`: camera-to-world transform (column vectors: right, down, forward, position)
- `w2c`: world-to-camera transform (inverse of c2w)

## Rectified Stereo Convention (SERV-CT)

SERV-CT provides rectified stereo where:
- Left camera frame = world frame (identity c2w/w2c)
- Right camera frame = world frame displaced by baseline along +X

## Depth Convention

- All internal depth tensors are in meters
- Invalid depth: NaN, Inf, ≤ 0, or outside [near, far]
