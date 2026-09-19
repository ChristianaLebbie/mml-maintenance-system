# Putting EIPS online: a step-by-step guide

This walks you through taking the Explainable Intervention Prioritization
System (EIPS) from "running on my laptop" to "a website I can open from my
phone, and my MML colleagues can log in to as well." It's written for
someone who has never done this before -- there's no step where you need
to already know what any of these tools are.

**Why these particular tools:** GitHub (free) stores your code online.
Streamlit Community Cloud (free) reads that code and runs it as a live
website. Neon (free) gives the live website a proper database, since the
free hosting doesn't reliably keep a local file (like the SQLite database
you use on your laptop) saved between restarts.

**A boundary worth naming up front:** creating these accounts and typing
in your own passwords is something only you should do -- it's your name,
your email, your credentials. Everything below is written as instructions
*for you to follow yourself*. If you'd rather do this together over a
screen-share, that works too; I just won't be the one entering anything
into a password field.

Budget about 45–60 minutes the first time, done in one sitting or spread
across a few sessions -- nothing here expires between steps.

---

## Before you start: what NOT to put online

Your app currently runs against a real Limble CMMS export from the mine
site. That data (and the local `database/predictive_maintenance.db`
file, which will end up holding real predictions and login passwords
once people use it) should never go into GitHub, even a private
repository -- repository history keeps every version forever, and
anyone ever added as a collaborator can see all of it.

The `.gitignore` file already in your project folder handles this
automatically: when you tell git to add your files (Step 2 below), it
will skip the data folders, the local database file, and any secrets
file. You don't need to remember to exclude them by hand -- just don't
override or delete `.gitignore`.

Once the site is live, you'll re-import your real dataset through the
app's own **Import Dataset** page (Step 7) -- straight into the new
hosted database, never through git.

---

## Step 1 — Create a GitHub account and a private repository

GitHub is where your code will live online so Streamlit Cloud can read
it.

1. Go to **github.com** and click **Sign up** (skip this if you already
   have an account).
2. Once signed in, click the **+** icon top-right → **New repository**.
3. Name it something like `eips` or `mml-maintenance-system`.
4. **Important:** set it to **Private**, not Public. This keeps the
   code (and later, anyone who tries to browse the repo) restricted to
   people you invite.
5. Leave everything else as default and click **Create repository**.
   GitHub will show you a page with setup commands -- you don't need
   those yet, just keep this browser tab open.

---

## Step 2 — Push your code to GitHub

This step happens on your own computer, in a terminal, from inside your
project folder (`Ama`). If you're not sure how to open a terminal there,
ask and I can talk you through it live on your machine -- typing the
commands themselves is fine for me to help with, since none of them
involve a password.

```
git init
git add .
git commit -m "Initial commit: EIPS"
git branch -M main
git remote add origin <the URL GitHub showed you in Step 1>
git push -u origin main
```

The `git push` step is the one place GitHub may ask you to sign in --
that's the one moment that's yours to complete.

If this is the first time using git on this machine, it may first ask
you to set a name and email (`git config --global user.name "..."` and
`user.email "..."`) -- any name/email is fine, it's just a label on your
commits.

---

## Step 3 — Create a free hosted database (Neon)

Streamlit Cloud's free tier doesn't reliably keep a local file saved
between restarts, so the SQLite file your laptop uses isn't suitable
once the app is hosted. Neon gives you a real, always-on Postgres
database for free, with no credit card required.

1. Go to **neon.tech** and sign up (signing up with your GitHub account
   is the fastest option, and keeps things to one fewer password to
   manage).
2. Create a new project — any name is fine (e.g. `eips-db`).
3. On the project dashboard, find the **Connection string** (sometimes
   labeled "Connection Details"). It looks like:
   ```
   postgresql://username:password@ep-something.neon.tech/dbname?sslmode=require
   ```
4. Copy that whole string somewhere safe for a moment — you'll paste it
   into Streamlit Cloud in the next step. Treat it like a password
   (because it has one embedded in it): don't paste it into chat, email,
   or anywhere public.

*(Supabase is a fine alternative if you'd prefer it — the free tier and
sign-up flow are similar, and it also gives you a `postgresql://`
connection string in the same shape. Either works with the app
unchanged.)*

---

## Step 4 — Create a Streamlit Community Cloud account and deploy

1. Go to **share.streamlit.io** and sign in with your GitHub account
   (this also grants Streamlit permission to read your repositories —
   normal and expected for this service).
2. Click **New app** (or **Create app**).
3. Choose your repository (`eips` / whatever you named it), branch
   `main`, and main file path `app.py`.
4. Before clicking Deploy, open **Advanced settings** → **Secrets**, and
   paste in:
   ```
   DATABASE_URL = "postgresql://username:password@ep-something.neon.tech/dbname?sslmode=require"
   ```
   using your real connection string from Step 3 (see
   `.streamlit/secrets.toml.example` in the project for the exact
   format). This box is private to your Streamlit account — it's not
   part of the code and never goes into GitHub.
5. Click **Deploy**. The first build takes a few minutes (it's
   installing everything in `requirements.txt`, including the machine
   learning libraries, so don't worry if it takes longer than a typical
   website).

When it finishes, you'll have a public URL like
`https://eips-something.streamlit.app` — this is the address that works
from any phone, laptop, or tablet, anywhere with internet, exactly as
you asked for.

---

## Step 5 — First visit: create your account

The very first time anyone opens that URL, the app will notice nobody
has an account yet and show a **"Set up the first account"** screen
instead of a normal login. Fill in your name, choose a username, and
choose a password — this becomes your personal sign-in, not a shared
password, and it's automatically given Admin standing.

From then on, that screen is a normal login form for anyone who already
has an account.

---

## Step 6 — Add your MML colleagues

Once you're signed in, go to **System Information** in the sidebar and
scroll to the **Accounts** section. There's a small form there to add a
new named account (name, username, password) for each colleague who
should have access. Everyone gets their own login — nobody has to share
yours.

---

## Step 7 — Bring your real data into the hosted database

Because the real MML dataset was deliberately kept out of GitHub (see
the note at the top), the hosted database starts empty. Sign in, go to
**Import Dataset**, and import your real CMMS export the same way you
would locally — it now writes straight into the Neon database instead of
the local SQLite file. The trained model that ships with the code is
already in place, so **Run Prediction** and the rest work right away
once a dataset is imported.

---

## After that: making changes

Whenever you (or I, working in a session with you) update the code
afterward, the update goes to your computer first, then you repeat just
the two `git` commands that send it to GitHub:

```
git add .
git commit -m "describe what changed"
git push
```

Streamlit Cloud watches your GitHub repository and automatically
rebuilds the live site within a minute or two of any push — no need to
redeploy by hand.

---

## Costs and limits, in plain terms

Both Streamlit Community Cloud and Neon's free tiers are genuinely free
(no credit card, no trial period that expires) — Streamlit Cloud's free
tier limits how much compute/memory an app can use and puts it to sleep
after a period with no visitors (it wakes up automatically, with a
~30-second delay, the next time someone opens the link). Neon's free
tier caps total storage at a few hundred MB and briefly "cold-starts"
after inactivity too, which is more than enough for a system holding
maintenance records for one site's equipment. If MML formally adopts
this beyond the thesis, that's the point to look at a paid tier for
speed and reliability — not before.
