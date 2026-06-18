# FabricCrypt launch sequence — timeline

All times in **UTC** unless noted. Target T=0 is **Tuesday 14:00 UTC**
(07:00 PT, 10:00 ET, 16:00 CEST). This is the time slot the viral
audit picked for X/Twitter peak engagement on a technical post.

---

## T-3 days — Saturday evening (UTC)

- [ ] Final read-through of `fabriccrypt_v3.md` (if ready) or v2.
- [ ] Build LaTeX in `launch_kit/arxiv_submission/`:
  ```
  pdflatex fabriccrypt && bibtex fabriccrypt
  pdflatex fabriccrypt && pdflatex fabriccrypt
  ```
- [ ] Tar the source: `tar -czf fabriccrypt_arxiv.tar.gz *.tex *.bib figures/`
- [ ] **Submit to arXiv** (Friday evening US → Monday morning UTC
      announcement). Categories: `cs.CR` primary, `cs.LG` secondary.
- [ ] License the preprint **CC-BY 4.0**.

## T-2 days — Sunday

- [ ] arXiv ID lands in your inbox; paste into:
      - `x_thread.md` tweet 10
      - `hn_post.md` body
      - `reddit_*.md` bodies
      - `press_release.md`
      - `CITATION.cff`
- [ ] Update GitHub README with arXiv badge.
- [ ] Verify https://github.com/Heigke/FabricCrypt is at the right
      commit; tag `v2.1` release.
- [ ] Final QA on `twitter_60s.mp4` and the 6-s transplant GIF.

## T-1 day — Monday

- [ ] Upload the long-form 3-minute video to YouTube as **Unlisted**.
- [ ] Schedule the unlisted → public transition for T-5 min via
      YouTube Studio.
- [ ] Send `casper_email.md` to Casper (and to two other AMD-laptop
      contacts).
- [ ] Five private DMs to named researchers (mention arXiv ID, ask
      for honest pre-publication critique):
      - Onur Mutlu (DRAM PUF lineage)
      - Yuval Yarom (DRAWNAPART)
      - Riccardo Paccagnella (Hertzbleed)
      - Daniel Genkin or Iakov Sanchez-Rola (clock-skew)
      - one named PCC / NVIDIA CC researcher
- [ ] Pre-draft your first HN comment in a local file; you'll paste
      it minutes after submission.

## T=0 — **Tuesday 14:00 UTC** (07:00 PT / 10:00 ET / 16:00 CEST)

- [ ] **T-5 min**: YouTube long-form goes public.
- [ ] **T=0**: Post tweet 1 of `x_thread.md` (60-s video attached).
- [ ] Drop the remaining 11 tweets in the thread back-to-back at
      ~30-second intervals (so the algorithm sees thread engagement).
- [ ] Pin tweet 1.
- [ ] Cross-post tweet 1 to LinkedIn and Bluesky.
- [ ] **T+30 min**: send the arXiv link to the five named researchers
      (if you didn't on Monday).
- [ ] **T+1h to T+4h**: actively answer X replies; do NOT post any
      other tweets (no jokes, no other content) until T+12h.

## T+48h — **Thursday 07:00 PT** (Wed 15:00 UTC)

- [ ] Submit to Hacker News using `hn_post.md` title.
- [ ] Within 2 minutes: paste the pre-drafted Q&A as the first
      author-comment.
- [ ] For the next 4 hours: answer every top-level technical
      comment within 15 minutes. **Do not argue with downvoters.**
      Address technical substance only.
- [ ] If HN front-page hits top-10, **delay the Reddit drops** to
      T+96h to avoid cannibalising attention.

## T+72h — **Friday US morning** (or T+96h if HN hot)

- [ ] **08:00 PT**: drop `reddit_ml.md` to r/MachineLearning.
- [ ] **09:30 PT**: drop `reddit_programming.md` to r/programming.
- [ ] **11:00 PT**: drop `reddit_crypto.md` to r/cryptography.
- [ ] Stagger so you can babysit each post for its first 90 minutes.

## T+1 week — Tuesday following

- [ ] Send `press_release.md` to:
      - **The Register** (techdesk@theregister.co.uk) — they love a
        good "vendor PKI is a racket" story.
      - **Ars Technica** (tips@arstechnica.com).
      - **MIT Technology Review** (editors@technologyreview.com).
      - **Wired** (tips@wired.com).
      - **Stack Diary** (matej@stackdiary.com).
      - **Phoronix** (michael@phoronix.com) — best chance of a
        thoughtful AMD/Linux-side write-up.
- [ ] Personalise the lede sentence per outlet.

## T+2 weeks

- [ ] Tally reproduction reports. If you have ≥3 chassis from the
      public, post a follow-up "FabricCrypt N=5" thread on X with
      the heatmap update.
- [ ] Submit to **USENIX Security '27** or **ACM CCS '27** (camera-ready
      with N≥5 included).

---

## Kill-switch — when to delay

Delay the public X drop by 24h if:

- arXiv preprint is not yet announced (preprint must precede thread).
- The 60-s video has artefacts that aren't fixed.
- An AMD-side security disclosure lands on the same Tuesday
  (we don't want to ride someone else's news cycle).
- Hacker News has a strong AMD/PSP-related front-page story already.

Delay the HN drop by 48h if:

- The X thread is still actively viral (>10k impressions on tweet 1
  and rising) — don't cannibalise.
- An obvious counter-example shows up in replies that needs a fix
  before HN sees it.
