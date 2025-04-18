import pandas as pd
import numpy as np

# 1) Build the 543‐point Holistic skeleton template
POSE_KPTS       = range(0, 17)
FACE_KPTS       = range(23, 91)
LEFT_HAND_KPTS  = range(91, 112)
RIGHT_HAND_KPTS = range(112,133)

skeleton = []
for typ, idxs in [
    ("pose", POSE_KPTS),
    ("face", FACE_KPTS),
    ("left_hand", LEFT_HAND_KPTS),
    ("right_hand", RIGHT_HAND_KPTS),
]:
    for li in idxs:
        skeleton.append((typ, li))
# This list is length 127, but we know the Holistic final layout is length 543:
#  0–32   pose (33 slots)
# 33–500  face (468)
#501–521  left hand (21)
#522–542  right hand (21)

# 2) Helper to map a frame’s detected rows → 543×3 array
def pad_to_543(frame_df):
    # frame_df has columns ['type','landmark_index','x','y','z']
    kp543 = np.zeros((543,3), dtype=np.float32)

    for _, row in frame_df.iterrows():
        typ = row['type']
        li  = int(row['landmark_index'])
        x,y,z = row[['x','y','z']].values

        if typ == "pose":
            out_idx = li
        elif typ == "face":
            out_idx = 33 + li
        elif typ == "left_hand":
            out_idx = 33 + 468 + li
        elif typ == "right_hand":
            out_idx = 33 + 468 + 21 + li
        else:
            continue

        kp543[out_idx] = [x,y,z]

    return kp543

# 3) Load your raw capture parquet (must have 'frame','type','landmark_index','x','y','z')
df = pd.read_parquet("3d_landmarks.parquet")

padded_frames = []
for frame_idx, group in df.groupby("frame"):
    # optionally sort/group if you like, but pad_to_543 only cares about type/index
    kp543 = pad_to_543(group)
    padded_frames.append(kp543)

# 4) Now re‐assemble into a DataFrame so we can write a Parquet
out_rows = []
for frame_idx, kp543 in enumerate(padded_frames, start=1):
    for idx_543, (typ, orig_li) in enumerate([
        ("pose", None)]*33 + [("face", None)]*468 + [("left_hand", None)]*21 + [("right_hand",None)]*21):
        # We only need typ and the local landmark_index = idx_543 - block_offset
        if idx_543 < 33:
            local_li = idx_543
            typ2     = "pose"
        elif idx_543 < 33+468:
            local_li = idx_543 - 33
            typ2     = "face"
        elif idx_543 < 33+468+21:
            local_li = idx_543 - (33+468)
            typ2     = "left_hand"
        else:
            local_li = idx_543 - (33+468+21)
            typ2     = "right_hand"

        x,y,z = kp543[idx_543]
        out_rows.append({
            "frame": frame_idx,
            "type": typ2,
            "landmark_index": local_li,
            "x": x, "y": y, "z": z
        })

out_df = pd.DataFrame(out_rows)
out_df.to_parquet("3d_landmarks_padded.parquet", index=False)
print("Saved padded 543‑point parquet:", out_df.shape)