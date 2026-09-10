# Start here

This folder runs a calculation for Harshit. It takes a few hours, and the computer
does all of it by itself.

You do not need to know anything about programming. Just follow these steps.

---

**Step 1.** Make sure this folder is fully unzipped, and that it is somewhere simple
like your Desktop. Do not run it from inside a zip file.

**Step 2.** Find the file called **`smoke_test`** and double-click it.
(On Windows it is `smoke_test.bat`. On a Mac or Linux it is `smoke_test.sh`.)

**Step 3.** A black window opens with text scrolling. Leave it alone. It takes about
two to five minutes the first time, because it downloads what it needs.

**Step 4.** Wait until it stops and read the last line.

- If it says **"Setup works  -  you can start the real run"** -> go to Step 5.
- If it says **"SOMETHING WENT WRONG"** -> copy the text it shows and send it to
  Harshit. Then stop here. See `IF_SOMETHING_GOES_WRONG.md` for the common fixes.

**Step 5.** Close that window. Now double-click **`run`**.
(On Windows, `run.bat`. On Mac or Linux, `run.sh`.)

**Step 6.** It will print a line like `Run 12 of 76  -  about 2 h 10 m left`. That is
all you need to watch.

**Expect roughly 3 to 5 hours in total  -  but please treat that as a rough guess.**
It was estimated from a short test on a different computer, not measured on yours, so
it could easily be half that or twice that depending on your machine. The countdown
in the window gets more accurate as it goes, because it uses your actual speed. If it
says something much larger than 5 hours, that is not a fault  -  just leave it running
overnight.

While it runs:
- You can use the computer normally  -  browse, email, watch things. It only uses part
  of the processor.
- You can leave it running overnight.
- **You can close the window whenever you like.** Nothing is lost. When you
  double-click `run` again, it carries on from where it stopped and skips everything
  it already did.

**Step 7.** When it finishes, the window prints **"ALL DONE"** and shows a file name.

**Step 8.** Send that one file back to Harshit. It is:

```
results/experiments.jsonl
```

Open the **`results`** folder inside this folder, and email or share the file called
**`experiments.jsonl`**. That is the only thing he needs.

---

### If you were also asked to add the second, optional dataset

Only if Harshit specifically asked. Open `GET_THE_OTHER_DATASET.md` and follow it.
If he did not ask, ignore that file completely  -  the main run does not need it, and
it will tell you so and carry on.

### If something breaks

Open `IF_SOMETHING_GOES_WRONG.md`.
