# Proposals to improve submission limits, grader reproducibility, and error visibility

- Topic ID: 693458
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693458
- Author: keymoon (@keymoon)
- Posted: 2026-04-21T04:35:52.020062800Z
- Votes: 8
- Total messages: 

## Body

This competition has a number of issues, largely stemming from the instability of `onnx-tool` and from inconsistencies between the participants' local environment and the grading environment. I'd like to make a few concrete requests that I think would meaningfully improve the experience. I don't believe any of these are unreasonable asks.

## 1. Add a final-judging stage and raise the daily submission limit

Last year's code-golf competition allowed up to 200 submissions per day. This one caps us at 5, and I guesed the reason is concern about overfitting to the test cases.

The problem is that 5 submissions per day is already more than enough to overfit on several of the tasks if someone is determined to do so. Meanwhile, for the rest of us, the cap just makes iteration painful and frankly, having to play that game at all isn't something I want to spend my time on.

I'd strongly encourage moving to a setup where **the final judging adds a large number of additional cases drawn from a publicly specified generation procedure**, on top of the existing public cases. If you're worried about collusion, you can publish the SHA-256 of the final test JSON up front and release the actual data after the competition ends that pins the organizers down without giving anything away.

With that in place, overfitting becomes a non-issue, the daily submission cap can safely be raised, and the barrier to entry drops significantly. This is a well-established pattern in long-running competitive programming contests, and it works well in practice.

Changes like this get harder to make the closer we get to the deadline, so if it's going to happen, sooner would be better.

## 2. A deadline for freezing the judging environment, and publishing that environment

When `onnx-tool` is updated, the set of viable optimization techniques can shift dramatically. I've already had the experience of discovering that an optimization I'd been relying on extensively was actually a bug in the tool, and having to re-optimize a large number of tasks as a result. Getting the rug pulled like that is demoralizing, and the work feels wasted.

I also currently have a submission locally that scores above the top of the leaderboard, but I haven't been able to get it past the grader. I've been careful about the small number of hidden random cases I crafted an input that also passes the ARC-GEN-generated random cases and yet. If I miss out on the longest-standing-leader prize over something like this, it'll be a real shame. And fundamentally, this comes back to the grading environment not being public.

So: **please commit to a date by which the grading environment will be frozen**, and announce it. That alone would make it much easier to plan time investment over the remaining weeks. Once it's frozen, **please publish a pinned `Dockerfile` and a `pyproject.toml` with an accompanying `uv.lock`** (or an equivalent) so participants can reproduce the exact environment locally.

I understand there's a reasonable concern that publishing the environment could surface security issues and enable bad actors. To address that, I'd suggest framing the release as "not the absolute final version, but the scoring logic is not expected to change from this point." That gets most of the benefit without the full commitment.

Either way, a clear freeze date **and** a published scoring environment would go a long way toward making this competition healthier for everyone.

## 3. Friendlier error messages

This one may be harder to act on, but it matters. Right now, if I submit a zip with 400 files and even one of them fails to load in the grading environment, the whole submission is rejected with no indication of which file failed or why. That leaves participants doing needle-in-a-haystack debugging, which burns submission budget and time.

If the grader simply reported which case failed and what the underlying error was, this kind of debugging would become tractable. And as a practical matter, this would also serve as a reasonable stopgap measure until the grading environment itself is published.

---

These are my suggestions for making the competition more fair, more reproducible, and more fun to participate in. I know you're busy, but I hope you'll consider them in whatever form makes sense. I genuinely want this competition to be as good as it can be that's the whole reason I'm writing this.

Thanks for reading.

## Comments (0)

(no comments)
