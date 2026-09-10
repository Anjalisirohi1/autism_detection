# If something goes wrong

Four things account for almost every problem. Find yours below.

Whatever happens, **nothing is ever lost**. Work that already finished stays
finished, and starting again picks up from where it stopped.

---

## 1. It says Python is not installed

**You will see:** a message beginning `PYTHON IS NOT INSTALLED`.

**Fix, on Windows:**

1. Go to <https://www.python.org/downloads/>
2. Click the big yellow **Download Python** button.
3. Open the file you just downloaded.
4. **On the very first screen, tick the box that says "Add python.exe to PATH".**
   This is the step people miss, and nothing works without it.
5. Click Install, wait for it to finish, then double-click `smoke_test` again.

**Fix, on a Mac:** download from the same page and install it the normal way, or open
the Terminal app and type `brew install python` if you already use Homebrew.

**Fix, on Linux:** open a terminal and run `sudo apt install python3 python3-venv`.

---

## 2. The window flashes open and shuts immediately

This normally means the folder is still zipped, or it is in a protected place.

**Fix:**

1. Move the whole folder to your **Desktop**.
2. If you have not unzipped it, right-click the zip file and choose
   **Extract All** (Windows) or double-click it (Mac), and use the folder that
   appears  -  not the zip.
3. Try again.

If it still flashes shut, open the folder, hold **Shift**, right-click in the empty
space, choose **Open PowerShell window here** or **Open in Terminal**, then type
`.\run.bat` (Windows) or `bash run.sh` (Mac/Linux) and press Enter. The window will
stay open and show the real message.

---

## 3. It says the install did not finish

**You will see:** `The install did not finish.`

This is almost always the internet connection, or a company network blocking the
download.

**Fix:**

1. Check you are online  -  load any website.
2. If you are on a work VPN, try turning it off, or try a home connection.
3. Double-click `run` again. It will retry.

If it fails three times with the same message, send Harshit the last twenty lines
from the window.

---

## 4. It says SOMETHING WENT WRONG partway through

**You will see:** a box saying `SOMETHING WENT WRONG`, then a short explanation and
a block of technical text.

**Fix:**

1. Copy **everything between the two dotted lines** and send it to Harshit.
2. Do not delete the folder  -  the finished work is still there and still useful.
3. If he asks you to, just double-click `run` again; it will resume.

---

## Other questions

**Can I stop it?**
Yes, at any time. Close the window, or press `Ctrl` and `C` together. Restarting
continues from where it stopped.

**Can I use the computer while it runs?**
Yes. It deliberately uses only part of the processor.

**Will it fill up my disk?**
No  -  it needs well under a gigabyte.

**Can I put the computer to sleep?**
Better not to. If it does sleep, the run pauses and continues when you wake it, but
closing the window and restarting later is cleaner.

**How do I know it is still working?**
The `Run X of Y` line updates each time a step finishes. Some steps take several
minutes, so a still screen is normal.

**It finished but I cannot find the file to send.**
Open the folder called `results` inside this folder. The file is
`experiments.jsonl`. If it is there and bigger than zero bytes, that is the one.

**Nothing here matches my problem.**
Send Harshit a photo or screenshot of the window. That is enough for him to work out.
