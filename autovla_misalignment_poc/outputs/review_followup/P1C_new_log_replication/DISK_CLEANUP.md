# P1-C disk cleanup log

Every deletion made by `scripts/review_followup/p1c/p1c_disk.py` is written here before it happens. Columns: UTC, path, files, bytes, reason, regeneration.

| UTC | path | files | bytes | reason | regeneration |
|---|---|---|---|---|---|
| 2026-10-08T15:36:00Z | `/root/VLA/autovla/dataset/nuplan/_p1c_tmp` | 0 | 0 | extraction scratch of shard 16 after moving its pilot_harness_check logs | scratch extraction dir; nothing to regenerate (re-created by the next fetch) |
| 2026-10-08T15:38:09Z | `/root/VLA/autovla/dataset/nuplan/_p1c_tmp` | 0 | 0 | extraction scratch of shard 20 after moving its pilot_harness_check logs | scratch extraction dir; nothing to regenerate (re-created by the next fetch) |
| 2026-10-08T18:23:03Z | `/root/VLA/autovla/dataset/nuplan/sensor_blobs/test/2021.09.29.19.02.14_veh-28_02911_03005/CAM_*/<504 files not used by selected scenes>` | 504 | 89557267 | shard 16: stage 1 done; images not used by any F/S-selected scene; kept 60 files needed by the F/S-selected scenes | `bash scripts/review_followup/p1c/p1c_driver.sh` fetch of shard 16 (CAM_F0/L1/R1 of this log) |
| 2026-10-08T18:26:44Z | `/root/VLA/autovla/dataset/nuplan/sensor_blobs/test/2021.10.06.08.16.17_veh-52_01590_01725/CAM_*/<756 files not used by selected scenes>` | 756 | 191918023 | shard 20: stage 1 done; images not used by any F/S-selected scene; kept 54 files needed by the F/S-selected scenes | `bash scripts/review_followup/p1c/p1c_driver.sh` fetch of shard 20 (CAM_F0/L1/R1 of this log) |
| 2026-10-08T18:59:30Z | `/root/VLA/autovla/dataset/nuplan/sensor_blobs/test/2021.09.29.19.02.14_veh-28_02911_03005` | 60 | 11080899 | pilot run finished (stage 2 complete); retained images of selected scenes no longer needed | re-fetch CAM_F0/L1/R1 of this log from navtest camera shard 16 (`p1c_driver.sh`) |
| 2026-10-08T18:59:30Z | `/root/VLA/autovla/dataset/nuplan/sensor_blobs/test/2021.10.06.08.16.17_veh-52_01590_01725` | 54 | 11952303 | pilot run finished (stage 2 complete); retained images of selected scenes no longer needed | re-fetch CAM_F0/L1/R1 of this log from navtest camera shard 20 (`p1c_driver.sh`) |
