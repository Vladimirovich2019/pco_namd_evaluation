# Excluded Cases in Metric Calculation

Cases whose prediction is **missing** (the model returned no usable response) or
**unrecognized** (the model answered, but the answer could not be resolved to one
of the four class labels) are excluded before metric calculation.

This applies to the treatment-response tasks only. All excluded cases belong to
the response-prediction settings; every experiment listed below was affected,
and no other experiment (MNV classification in any input setting, response
prediction in any other model/setting) contained a missing or unrecognized
prediction.

## Excluded case counts

Counts are per run, out of 101 response cases.

| Model | Setting | Run 1 | Run 2 | Run 3 |
|-------|---------|-------|-------|-------|
| GPT | Images Only | 1 | 1 | 2 |
| GPT | Combination | 1 | 1 | 1 |
| Qwen VL Max | Images Only | 16 | – | – |

Legend: **Images Only** = response prediction from CFP and OCT alone;
**Combination** = response prediction from images together with the baseline
biomarker measurements.

All remaining experiments — every MNV classification task, and response
prediction for Gemini 3.1 Pro in all three settings as well as Qwen VL Max in
the `w/o Images` and `Combination` settings — yielded a recognized prediction for
every case, so their metric denominators are the full case count.

## Effect on reported metrics

Excluding these cases changes the denominator of the affected runs only: the
number of cases entering each metric is the total number of cases in that
experiment minus the count in the table above. The exclusion is applied
uniformly to every model and every task by the same code path, and the number of
excluded cases is reported alongside the metrics whenever the reproduction script
is run, so that the denominators behind each reported value are recoverable.
