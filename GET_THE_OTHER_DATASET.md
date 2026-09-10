# Optional: adding the second dataset

**Only do this if Harshit specifically asked you to.** The main run works perfectly
without it and will simply say the optional dataset is not present, then carry on.

The reason you have to fetch this one yourself, rather than it being included, is a
licensing one: the people who published it did not attach a licence, so it cannot be
passed on second-hand. Downloading it directly from them is fine.

---

## What to do

**Step 1.** Open this link in your browser:

<https://github.com/pavanravva/Enhanced-MMASD>

**Step 2.** On that page, scroll to the section headed **"Code and Data"** and click
the Google Drive link there.

**Step 3.** In the Google Drive folder, find the folder named **3D-Skeleton** and
download it. It is a few hundred megabytes and downloads as a zip file.

**Step 4.** Unzip what you downloaded.

**Step 5.** Inside this bundle, create a folder path exactly like this:

```
data/mmasd_plus/raw/
```

**Step 6.** Put the unzipped folders (the ones with names like `Arm_Swing`,
`Drumming`, `Tree_Pose`) inside `data/mmasd_plus/raw/`.

You should end up with something that looks like:

```
data/mmasd_plus/raw/Arm_Swing/processed_....csv
data/mmasd_plus/raw/Drumming/processed_....csv
...
```

**Step 7.** Double-click `run` as normal. It will notice the extra data, tell you it
found it, and include it automatically.

---

## Notes

- If you get the folder structure slightly wrong, the run will just say the optional
  dataset is not present and continue with the main work. Nothing breaks.
- This second part adds roughly one to two more hours.
- You can add it later, after the main run has already finished. Rerunning skips
  everything that is already done and only runs the new part.
- Do not send this downloaded data back to Harshit  -  he only needs
  `results/experiments.jsonl`, same as before.
