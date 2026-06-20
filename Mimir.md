# 🎤 Pycon Talk — Mimir: AI Image Detection System

---

## 🏷️ Title Options

### Punchy / Memorable
1. **"Don't Believe Your Eyes: Building a Python AI Image Detective"**
2. **"Ctrl+Fake: How I Built a Bot to Fight AI Misinformation"**
3. **"Seeing Through the Lie: AI Image Detection in Python"**
4. **"The Forgery Hunter: Building an AI Provenance Tool in Python"**
5. **"Python vs. the Deepfake: A Forensics Story"**

### Technical / Conference-Forward
6. **"From Pixels to Provenance: A Forensic AI Detection Pipeline"**
7. **"Teaching Python to Spot What Your Eyes Cannot"**
8. **"Stacking the Evidence: A Multi-Signal AI Image Classifier"**
9. **"Mimir: A Production-Grade AI Image Provenance System"**

### Bot / Activist Angle
10. **"Fighting Fake News, One WhatsApp Image at a Time"**
11. **"A Bot That Fights Fake News (And Learns From Its Mistakes)"**
12. **"Arming Journalists with Python: An AI Detection Bot for the Field"**
13. **"The Watchdog Bot: Detecting AI-Generated Propaganda with Python"**

### Poetic / Evocative
14. **"Dead Pixels Tell No Tales — But Python Does"**
15. **"What Hath AI Wrought? A Python Detective Story"**

---

## 🗂️ Track / Focus Area Options

| Track | Why It Fits |
|---|---|
| **AI & Machine Learning** | ViT, Hugging Face Transformers, RL/Contextual Bandit, ELA forensics |
| **Security & Privacy** | Misinformation detection, provenance verification, media forensics |
| **Data Science & Society** | Real-world impact of AI-generated images on journalism and public trust |
| **Python in the Wild** | Full-stack Python: Streamlit + FastAPI + SQLite + RL agent |
| **Ethics in AI** | The arms race between AI generators and detectors; responsible AI |
| **Open Source Tools** | Building practical forensics pipelines with open-source Python libraries |

**Best fit:** `AI & Machine Learning` + `Ethics in AI`
**Alternative (wider audience):** `Security & Privacy` + `Python in the Wild`

---

## 📋 Synopsis Options

### Synopsis A — Story/Impact Focus *(Recommended)*
> In Africa, data is expensive. Yet WhatsApp is everywhere — on the cheapest smartphones,
> in the smallest villages, in newsrooms and family group chats alike. It has become the
> primary way millions of people share news, information, and truth.
>
> That same accessibility makes it the perfect vehicle for fake news. With modern AI tools,
> anyone can generate a convincing fake image in seconds and share it to thousands of people
> before a single journalist has had the chance to verify it. The damage spreads faster than
> the correction ever will.
>
> This talk is about what I built to fight back.
>
> **Mimir** is an open-source AI image detection system built entirely in Python — designed
> to be accessible to journalists, researchers, and everyday fact-checkers. It combines five
> detection layers: metadata & C2PA provenance extraction, invisible watermark decoding,
> perceptual hashing, a fine-tuned Vision Transformer from Hugging Face, and forensic Error
> Level Analysis. A built-in Reinforcement Learning agent means the system gets smarter
> every time a human corrects it.
>
> The end goal: a WhatsApp bot anyone can message with a suspicious image and get a clear,
> evidence-based verdict — no technical knowledge required.
>
> No prior forensics experience needed. If you write Python and care about truth, this talk
> is for you.

---

### Synopsis B — Technical Focus
> Fake images are cheap to generate and expensive to detect — and most of the tools that
> exist require expert knowledge to use. This talk presents **Mimir**, a fully open-source,
> production-quality AI image detection system built in Python, designed from the ground up
> to be accessible.
>
> We walk through a five-layer forensics pipeline: EXIF & C2PA metadata extraction,
> invisible watermark decoding, perceptual hashing, a fine-tuned Vision Transformer (ViT)
> from Hugging Face, and Error Level Analysis (ELA). We then add a **Contextual Bandit RL
> agent** that dynamically re-weights each detection layer based on real-world feedback —
> so the system improves with every use.
>
> Finally, we discuss packaging this into a WhatsApp bot via FastAPI — turning a research
> tool into something journalists can use directly in the field on the devices they already own.

---

### Synopsis C — Short CFP Abstract *(100 words)*
> Data is expensive. WhatsApp is everywhere. And modern AI makes it trivially easy to
> generate convincing fake images at scale. **Mimir** is an open-source Python system I
> built to fight back — making AI image detection accessible to journalists and fact-checkers
> who need it most. It combines a five-layer forensics pipeline (metadata, watermarks,
> perceptual hashing, a Hugging Face Vision Transformer, and ELA) with a Contextual Bandit
> RL agent that learns from human feedback. The goal: a WhatsApp bot anyone can send a
> suspicious image to and receive a clear, evidence-based verdict — no technical skills required.

---

## 🎯 Top 3 Recommended Titles

| # | Title | Why |
|---|---|---|
| 🥇 | **"Fighting Fake News, One WhatsApp Image at a Time"** | Immediately frames the problem and the solution — strong emotional hook |
| 🥈 | **"Don't Believe Your Eyes: Building a Python AI Image Detective"** | Clear, broad appeal, works for all experience levels |
| 🥉 | **"Ctrl+Fake: How I Built a Bot to Fight AI Misinformation"** | Punchy, techy, memorable on a conference programme |

**Best synopsis to submit:** Synopsis A (story-driven) for the full description, Synopsis C (short abstract) for the CFP word-limited field.

---

## ⏱️ Suggested Talk Duration
- **30 min** — Pipeline architecture + RL agent overview
- **45 min** — Adds live demo: scan a real AI image vs. a human photo on stage
- **60 min** — Deep dives into ViT internals + RL feedback loop + WhatsApp bot integration


---

## 🏷️ Title Options

### Punchy / Memorable
1. **"The Journey of a Thousand Fake Images Begins with a Single Pixel"**
2. **"Don't Believe Your Eyes: Building a Python AI Image Detective"**
3. **"Ctrl+Fake: How I Built a Bot to Fight AI-Generated Misinformation"**
4. **"Python vs. the Deepfake: A Forensics Story"**
5. **"Seeing Through the Lie: A Multi-Layer Approach to AI Image Detection"**
6. **"The Forgery Hunter: Building an AI Provenance Tool in Python"**
7. **"One Image, Five Lies: Unmasking AI-Generated Media with Python"**

### Technical / Conference-Forward
8. **"Multi-Layer AI Image Forensics with Python: From ELA to Vision Transformers"**
9. **"Mimir: Building a Production-Grade AI Provenance Pipeline in Python"**
10. **"From Pixels to Provenance: A Forensic AI Detection System in Python"**
11. **"Teaching Python to Spot What Your Eyes Cannot"**
12. **"Reinforcement Learning Meets Forensics: A Self-Improving AI Detector"**
13. **"Stacking the Evidence: How to Build a Multi-Signal AI Image Classifier"**

### Bot / Activist Angle
14. **"A Bot That Fights Fake News (And Learns From Its Mistakes)"**
15. **"Fighting Fake News, One WhatsApp Image at a Time"**
16. **"The Misinformation Pipeline: Building Tools to Break It"**
17. **"Arming Journalists with Python: An AI Detection Bot for the Field"**
18. **"The Watchdog Bot: Using Python to Detect AI-Generated Propaganda"**
19. **"From Newsroom to Node: How Python Can Fight Visual Misinformation"**

### Poetic / Evocative
20. **"What Hath AI Wrought? A Python Detective Story"**
21. **"The Oracle's Eye: How Python Sees What Humans Miss"**
22. **"Mimir's Eye: Named After the Norse God of Knowledge for a Reason"**
23. **"Dead Pixels Tell No Tales — But Python Does"**
24. **"In a World of Fake Images, the Forensic Bot Is King"**

---

## 🗂️ Track / Focus Area Options

| Track | Why It Fits |
|---|---|
| **AI & Machine Learning** | Core use of ViT, Hugging Face Transformers, RL/Contextual Bandit, ELA forensics |
| **Security & Privacy** | Misinformation detection, provenance verification, media forensics |
| **Data Science & Society** | Real-world impact of AI-generated images on journalism and public trust |
| **Python in the Wild** | Full-stack Python: Streamlit dashboard + FastAPI + SQLite + RL agent |
| **Ethics in AI** | The arms race between AI generators and AI detectors; responsible AI tools |
| **Open Source Tools** | Building practical, extensible forensics pipelines with open-source Python libraries |

**Best fit (combined pitch):** `AI & Machine Learning` + `Ethics in AI`
**Alternative (wider audience):** `Security & Privacy` + `Python in the Wild`

---

## 📋 Synopsis Options

### Synopsis A — Technical Focus
> Fake images are cheap to make and expensive to detect. In this talk, we walk through the
> architecture of **Mimir** — a fully open-source, production-quality AI image provenance
> detection system built entirely in Python.
>
> We explore a five-layer forensics pipeline: EXIF & C2PA metadata extraction, invisible
> watermark decoding, perceptual hashing, a fine-tuned Vision Transformer (ViT) from
> Hugging Face, and Error Level Analysis (ELA). We then go beyond static detection by
> introducing a **Contextual Bandit RL agent** that dynamically re-weights each detection
> layer based on real-world user feedback — so the system gets smarter over time.
>
> Finally, we discuss integrating this pipeline into a WhatsApp bot via FastAPI, turning a
> research tool into something journalists and investigators can use directly in the field.
>
> **You will leave knowing:** how to combine classical forensics with modern deep learning,
> why ELA fails on AI images (and what to do instead), and how to build a self-improving
> detection pipeline with reinforcement learning.

---

### Synopsis B — Story/Impact Focus
> In 2024, a single AI-generated image shared on WhatsApp sparked a public panic in three
> countries before it was debunked — 72 hours too late. The tools to detect it existed. No
> one had packaged them into something a journalist could actually use.
>
> This talk tells the story of building **Mimir** — a Python-powered AI image detection
> bot designed for investigators and researchers in the field. We cover the full journey:
> why traditional image forensics tools fail against modern diffusion models, how we built
> a five-layer detection pipeline combining metadata, watermarks, perceptual hashing, and
> a fine-tuned Vision Transformer, and how we made it self-improving using reinforcement
> learning so that every correction from a human expert makes the system smarter.
>
> No previous forensics experience required. If you write Python and care about truth, this
> talk is for you.

---

### Synopsis C — Short Abstract (for CFP forms, 100 words)
> AI-generated images are undermining public trust at scale. This talk presents **Mimir**,
> a production-grade Python system that detects AI-generated imagery using a five-layer
> forensics pipeline: metadata & C2PA extraction, invisible watermark decoding, perceptual
> hashing, a Hugging Face Vision Transformer, and Error Level Analysis. We go further by
> adding a **Contextual Bandit RL agent** that continuously re-weights each detection layer
> based on expert feedback, making the system self-improving. Attendees will leave with
> practical knowledge of image forensics, Hugging Face integration, Streamlit/FastAPI
> architecture, and applied reinforcement learning — all in Python.

---

## 🎯 My Top 3 Recommendations

| # | Title | Why |
|---|---|---|
| 🥇 | **"Don't Believe Your Eyes: Building a Python AI Image Detective"** | Instantly clear, broad appeal, works for all experience levels |
| 🥈 | **"Ctrl+Fake: How I Built a Bot to Fight AI-Generated Misinformation"** | Memorable, techy, punchy — great for conference posters |
| 🥉 | **"Fighting Fake News, One WhatsApp Image at a Time"** | Most relatable to a general audience, strong geographic/social angle |

**Best synopsis to submit:** Synopsis B (story-driven) for the abstract, Synopsis A (technical) for the full description.

---

## ⏱️ Suggested Talk Duration
- **30 minutes** — Covers the pipeline architecture + RL agent overview
- **45 minutes** — Adds live demo: scan a real AI image vs. human photo on stage
- **Full 60 minutes** — Deep dives into ViT internals + RL feedback loop + WhatsApp bot integration
