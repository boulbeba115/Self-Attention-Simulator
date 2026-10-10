# LLM Simulator: see how AI picks its next word

An interactive, step-by-step simulator that shows what happens inside a language model when it predicts the next word. Type the start of a sentence, press run, and watch a miniature transformer work through tokens, embeddings, self-attention, the feed-forward network, word scores and sampling, with every matrix and every calculation visible.

**[▶ Try it live](https://boulbeba115.github.io/Self-Attention-Simulator/)**

> Replace the link above with your GitHub Pages address once the site is published.

---

## Features

- **Twelve animated steps per word.** Each stage of the transformer gets its own slide with a short, plain-language explanation.
- **Every number is clickable.** Tap any cell in a matrix to see exactly how it was calculated, with its inputs highlighted.
- **Live attention on the start page.** Your words appear as tiles joined by arcs: the thicker the arc, the more one word listens to another. Point at a word to see what it listens to, and watch the model's guess change as you type.
- **Explanation bubbles.** Terms like *query*, *softmax* and *LayerNorm* open a simple definition when you hover or tap them.
- **Math refresher.** Short interactive lessons on the dot product, matrix multiplication (a 2×2 example) and softmax, for anyone who has forgotten the maths.
- **Adjustable model.** A settings pop-up changes the number of words to predict, vector size, number of layers, temperature, top-k, sampling vs. greedy choice, and the random seed, with a preview of what each change does. It also opens while you are stepping, and restarts the run with the new settings.
- **No fixed length.** When a run ends, “Keep writing” adds more words for as long as you like. The model looks at the last 24 words, its context window.
- **One navigation bar everywhere.** The top bar stays while you step through the model, and its links take you back to the home page.
- **Autoplay and keyboard control.** Use ← and → to move between steps, and Space to play or pause.
- **Works everywhere.** It's a handful of static files with no build step and no server, and it works on desktop, tablet and mobile.

## The twelve steps

| # | Step | What happens |
|---|------|--------------|
| 1 | Tokens | The sentence is split into tokens, and each token gets its ID in the vocabulary. |
| 2 | Embeddings | Each ID is turned into a vector of numbers that represents the word's meaning. |
| 3 | Positions | Sine and cosine waves are added so the model knows the word order. |
| 4 | Query, key, value | Each word vector is multiplied by three weight matrices. |
| 5 | Scores | Every query is compared with every key (Q·Kᵀ / √d). |
| 6 | Mask | Future words are hidden by setting their scores to −∞. |
| 7 | Attention | Softmax turns each row of scores into percentages. |
| 8 | Blend | Each word mixes the value vectors of the words it attends to. |
| 9 | Add & normalize | The input is added back (the residual connection), then LayerNorm rescales each row. |
| 10 | Feed-forward | A small neural network with ReLU processes each word on its own. |
| 11 | Word scores | Every word in the vocabulary gets a score (a logit). |
| 12 | Choose | Temperature, top-k and a random draw pick the next word. |

The chosen word is added to the sentence, and the whole process runs again. This loop is called autoregressive generation.

## How realistic is it?

The maths is real: matrix multiplications, scaled dot-product attention, causal masking, softmax, residual connections, LayerNorm, ReLU and temperature sampling all work exactly as they do in GPT-style models, only at a tiny size.

Two things are simplified so the model can run instantly in a browser without training:

- **Word embeddings are learned from a built-in text** of about 53,000 sentences and 5,300 different words, in English with a little French. About 2,750 sentences are hand-written and cover everyday talk, animals, countries and capitals, jobs, food, numbers, days, opposites, and how language models work. The other 50,000 are short story sentences sampled from the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. This is done by factorising a word co-occurrence matrix, the idea behind GloVe and word2vec. Words used in similar ways, such as *cat* and *dog*, end up with similar vectors.
- **The network weights are not trained by gradient descent.** To get sensible predictions, the final scores add a "training-text" signal: phrase statistics counted from the built-in text (an n-gram model over phrases of up to four words, with back-off). The app labels this openly. Set **Training-text strength** to 0 to see what the raw, untrained network produces.

Real models use sub-word tokens, vectors with thousands of numbers, dozens of layers and multiple attention heads. This simulator uses whole words, 4 to 8 numbers per vector, 1 to 3 layers and a single head, so that every number fits on screen.

## Run it locally

No installation is needed. Download the repository and open `index.html` in any modern browser (Chrome, Edge, Firefox or Safari).

## Deploy on GitHub Pages (free)

1. Put the files at the root of a **public** repository.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/ (root)`, then click **Save**.
4. After a minute or two, the site is live at `https://YOUR-USERNAME.github.io/YOUR-REPO/`.

## Tech

- Plain HTML, CSS and JavaScript, with no frameworks and no dependencies.
- All computation runs client-side, and no data leaves the browser.
- Fonts: the system font (SF Pro) on Apple devices, and [Inter](https://fonts.google.com/specimen/Inter) from Google Fonts elsewhere. Without an internet connection, the page falls back to your system fonts.

## Project structure

```
.
├── index.html          # markup
├── styles.css          # styles
├── corpus.js           # the training text: hand-written sentences and fact tables
├── corpus-stories.js   # 50,000 story sentences sampled from TinyStories
├── tools/
│   └── build-stories.py   # rebuilds corpus-stories.js from the dataset
├── script.js           # model, slides and animations
├── assets/             # logo
└── README.md
```

## Customising the training text

The model's knowledge lives in `corpus.js`: blocks of hand-written sentences, one per line, plus small fact tables (animals, countries, jobs, and so on) that are expanded into sentences. Add sentences or table rows there and the vocabulary, embeddings and phrase statistics update automatically the next time the page loads.

The story sentences live in `corpus-stories.js`, which is generated. To make it larger or smaller, download `TinyStoriesV2-GPT4-valid.txt` from the dataset page and run:

```
python tools/build-stories.py path/to/TinyStoriesV2-GPT4-valid.txt --sentences 50000 --vocab 6000
```

More sentences teach the model more, but the page gets heavier: at 50,000 sentences the file is about 2 MB (0.6 MB compressed). Delete the file's `<script>` tag from `index.html` to run on the hand-written text alone.

## Data credit and licence

`corpus-stories.js` is a sample of the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset by Ronen Eldan and Yuanzhi Li ([paper](https://arxiv.org/abs/2305.07759)). The dataset and this sample are licensed under the [Community Data License Agreement – Sharing 1.0](https://cdla.dev/sharing-1-0/). The sample was made by splitting the stories into sentences, lower-casing them, removing punctuation, and keeping sentences of 4 to 14 common words.
