# Why NER Was Never the Bottleneck — Verbatim Gold and Intent Errors Explained

Date: 2026-10-13
Scope: plain-language elaboration of Round 11's cross-cutting finding (external NER round): entity
*detection* is solved, gold answers quote users verbatim, and the leftover errors are judgment calls
requiring intent understanding — plus what that implies for improving the method. Companion to
`notes/2026-10-13-round11-plain-language-summary.md`.

## 1. Two different skills: *finding words* vs *understanding requests*

Think of a waiter in a restaurant. Skill A: **hearing** — catching that the customer said "grilled
salmon, no lemon, near the window". Skill B: **understanding** — knowing that "no lemon" is an
instruction to the kitchen, "near the window" is for the seating host, and "salmon" is the actual order.

Entity recognition (NER — GLiNER, Flair, Stanza) is pure Skill A. It's a **highlighter pen**: it marks
"this is a city", "this is a street". Our experiments showed Skill A was never the problem:

- We ran **four completely independent NER systems** (wikineural, GLiNER, Flair×3) over the test data.
  In English, all four found exactly **71.8%** of the gold values. In French, all four hit exactly
  **81.0%**. When different tools built on different technology land on the *identical* number, the
  limit isn't the tools — it's the task itself.
- The 20-30% they all miss isn't cities at all — it's things like `"parking"`, `"charging station"`,
  `"hotel on Maple Street"`. Search phrases and categories simply **aren't dictionary entities**. No
  better dictionary will find them, because they aren't in any dictionary — they're free speech.

**Analogy:** we brought in better and better metal detectors, then discovered the field wasn't full of
metal — it was full of plastic. The detector's accuracy stopped mattering.

## 2. Why "correcting" names *breaks* things: the transcript exam

Our benchmark's gold answers work like this: the grader holds a **verbatim transcript of what the user
actually said**. If the user said "Stockholm", gold says `Stockholm` — even in a Polish sentence, even
though the official Polish name is "Sztokholm".

External knowledge bases (GeoNames, DBpedia, Wikidata) have exactly one superpower: knowing the
**official, canonical name** of every place. So we tried using them to "fix" our model's outputs.
Result on training data: **111 of 114 corrections were harmful**. `Krakow` → `Cracovia` (the Latin
exonym — technically "correct", now mismatched). `Stockholm` → `Sztokholm`. Every fix moved the answer
*away* from the transcript.

**Analogy:** it's like an exam graded by comparing your answer to a stenographer's record of the
customer's exact words. If you "helpfully" correct their spelling to dictionary spelling, you get marked
*wrong*. The exam rewards **echoing**, not **correcting**.

The flip side explains why our small homemade alias table (v1) works: when the model output IS Polish
grammar noise ("Krakowa" — the inflected form), the fix needed is "undo the grammar of *this same
sentence*" — a tiny table of inflections observed in *our* training data handles that perfectly
(Polish +12.9 F1). The big external dictionary either never fires, or fires dangerously. That's why v2
ended up as a never-firing safety net.

## 3. The leftover errors are *judgment calls*, not lookups

After all mechanical fixes, two error families dominate, and both are Skill B:

### (a) The q-boundary problem — cropping the photo

For "find parking", the model must output the search query `q`. Given *"parking near our hotel on
Maple Street"*, is `q` = "parking"? "parking near our hotel"? "hotel on Maple Street"? That's a
**cropping decision**. GLiNER can mark "Maple Street" as a street — but how much context to keep is
intent. We proved this painfully: our arbitration trimmed "our hotel on Maple Street" → "Maple Street"
(**correct**, gold agreed) and "Detroit, Michigan" → "Detroit" (**wrong**, gold wanted both). And the
killer: "centrum Portland" → gold wanted just "Portland", but "centro de Springfield" → gold wanted the
whole phrase. *Same grammatical pattern, opposite gold decisions.* To a span detector these are
geometrically identical — the difference is only visible if you understand what each user was trying to
do. Result: arbitration broke 9 answers for every 4 it fixed.

**Analogy:** a photo-cropping assistant. The face detector finds the face flawlessly — but "crop tight
to the face" vs "keep the scenery" depends on what the photo is *for*. Two identical-looking photos, two
different right answers.

### (b) The avoid_features problem — the blank receipt

User says *"route without highways"* / *"sin usar autopistas"*. Should the answer record
`avoid_features: motorway`? Our model says yes — **semantically it's right**. But in ~24 of 33 such
cases, the gold answer is `null`, because the human annotator simply didn't fill the field. The model
gets penalized for understanding the user.

**Analogy:** a waiter carefully writes "no onions" on the order — but the official receipt template left
the extras field blank. The waiter is right, the receipt is wrong, and no dictionary on Earth fixes
receipts.

## 4. So what actually improves the method now?

The insight inverts the roadmap: stop buying better highlighter pens; invest in **judgment** and in
**the answer key**. Concretely, in order of cost:

1. **Fix the receipts (gold hygiene)** — the ~24 "no onions / blank receipt" rows are the same
   label-noise class we already fixed twice (round-4 `lang` slot, round-7 benchmark v2). Cleaning them
   converts "impossible" errors into learnable ones. Zero inference cost, likely the cheapest F1 in the
   repo.
2. **Ask a senior clerk only when the junior hesitates (cascade)** — we already built uncertainty
   detectors and a 3-model voting ensemble in earlier rounds. Use them to flag the *judgment-call rows*
   (q-boundary, enum-vs-null) and route only those ~10-15% to a bigger model. This is precisely the
   Skill-B lane: spend the intelligence budget where intent, not memory, is the issue.
3. **Teach judgment with contrastive training data** — e.g. augmentation pairs where the *same* place
   phrase is gold-inclusive in one intent ("navigate to centro de Springfield") and gold-exclusive in
   another ("find parking near the centro in Springfield"). Small models learn judgment only from
   examples that contrast it.
4. **A slightly bigger decoder** — intent understanding is the textbook thing model scale buys (a 3B
   retrain of the same recipe). Most expensive, most reliable.

**One-liner:** we were polishing the dictionary when the exam was an essay. Detection is solved; the
remaining points live in intent understanding and in the answer key itself.
