# 8008.14 authority audit

## Outcome

Read-only comparison reproduced **37 exact payload changes** from `submission_base_8006.61.zip` to `submission_base_8008.14.zip`. All changed payloads are competition-scorer correct and strictly cheaper. The summed projected gain is **+1.532830682633**, consistent with the displayed checkpoint increase `8008.14 - 8006.61 = +1.53` (rounding residual +0.002830682633).

Both ZIPs have 400 unique expected members, are CRC-clean, preserve identical member order, and have Conv-family bias UB count zero across all 400 models.

## Authority hashes

- 8006.61 SHA-256: `9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`; MD5: `c90b23f514fe47c36f1d032bd0924662`
- 8008.14 SHA-256: `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`; MD5: `db4da5cc59186b26572a380725bc2fdf`

## Changed payloads

| task | old SHA | new SHA | old cost | new cost | projected gain |
|---:|---|---|---:|---:|---:|
| 013 | `01736037ac2a` | `f72854de968a` | 636 | 357 | +0.577462782 |
| 019 | `5877c02c963d` | `e8d7c5ca20fa` | 536 | 535 | +0.001867414 |
| 035 | `f898a27ddd8e` | `82b9e298e974` | 545 | 544 | +0.001836548 |
| 037 | `ed757e114a44` | `df9298f3b9e8` | 374 | 320 | +0.155934802 |
| 046 | `71aae8143863` | `fb649383229d` | 631 | 627 | +0.006359322 |
| 062 | `ab49737eda12` | `6767dbf75899` | 465 | 463 | +0.004310352 |
| 066 | `27098b37cb13` | `bb8cebc8d71d` | 677 | 562 | +0.186169423 |
| 089 | `cf7262764173` | `89183f12515c` | 1349 | 1340 | +0.006693963 |
| 096 | `d026067708b4` | `97f05f8495c7` | 1128 | 1123 | +0.004442477 |
| 107 | `b47fbb5396e8` | `39e937f3065e` | 708 | 664 | +0.064161944 |
| 117 | `e8dee03b3c5f` | `042e3ee0976a` | 606 | 605 | +0.001651528 |
| 125 | `cbab7604bcc9` | `c30ac7a079a4` | 1050 | 1045 | +0.004773279 |
| 138 | `8a6400d6fa69` | `55e71aec7157` | 2729 | 2705 | +0.008833329 |
| 156 | `b5457e5da157` | `e8b10010b50a` | 556 | 499 | +0.108162198 |
| 157 | `fcafb8af5728` | `a1254f261940` | 853 | 849 | +0.004700361 |
| 165 | `2e1af6681882` | `d6d40c11204c` | 592 | 587 | +0.008481815 |
| 168 | `dcf6a0cc845c` | `642cba5c350b` | 416 | 415 | +0.002406740 |
| 170 | `756f6d2cd27b` | `e5fea4c41d22` | 387 | 384 | +0.007782140 |
| 191 | `109928c1f7ec` | `76795962c336` | 3444 | 3436 | +0.002325582 |
| 209 | `9d0c21971843` | `80c19164133e` | 2218 | 2087 | +0.060878261 |
| 245 | `e22d0d661df9` | `228b6ad9f245` | 387 | 385 | +0.005181359 |
| 268 | `f77d0468fcf6` | `4c8ec91a517e` | 422 | 420 | +0.004750603 |
| 270 | `8272254bf947` | `0d848124abaf` | 594 | 587 | +0.011854500 |
| 284 | `d8f60072c5f0` | `0d03efd73a59` | 518 | 517 | +0.001932368 |
| 308 | `06d6ae77684c` | `fc845e9edee0` | 434 | 433 | +0.002306806 |
| 310 | `f7ad4fb86c5a` | `4eed21efedf2` | 566 | 501 | +0.121987977 |
| 338 | `edcac049616e` | `09e8436ab305` | 426 | 406 | +0.048086187 |
| 361 | `bb6b069c1f45` | `d606fcf6e115` | 858 | 844 | +0.016451605 |
| 365 | `cb9eed3b7eb8` | `85d63fa65d51` | 1369 | 1355 | +0.010279092 |
| 370 | `9aa470ae8acc` | `513c0b40056f` | 954 | 944 | +0.010537505 |
| 378 | `3e66557d91ea` | `a5cf4a598239` | 525 | 522 | +0.005730675 |
| 182 | `605285fc5c04` | `625b31492d91` | 951 | 949 | +0.002105264 |
| 044 | `49a1d5459ca9` | `12b6414193b8` | 1086 | 1076 | +0.009250760 |
| 069 | `87fbae86b65c` | `8a7bdff92bed` | 541 | 524 | +0.031927595 |
| 354 | `c86ec60a3cf1` | `4ba066a4ccb0` | 537 | 536 | +0.001863933 |
| 363 | `aec5b5333bb9` | `5daecf63ed4b` | 513 | 512 | +0.001951220 |
| 014 | `ec765b010bbb` | `15a7de7d7ad0` | 370 | 360 | +0.027398974 |

## Known-black controls

All five requested black controls retain the exact 8006.61 payload and do not appear in the changed-task set.

| task | payload SHA-256 | old/new identical | changed list |
|---:|---|---|---|
| 023 | `bd242d29ab9514b2432dce31e6df28dd67f00bf1bdcb54c8a00f28614f974fb0` | true | false |
| 198 | `4e37cca3fc86cd4781a9b1f55c080f13962273e803c4c45d6dda99f74ba95283` | true | false |
| 201 | `fb28f6065fbac760fd5a9e40d00af44eb5128c3d76676e852e62baddb574beda` | true | false |
| 208 | `6c9bad970152f9380f07954878876c474dda51752586a200e4a911105fa4d016` | true | false |
| 396 | `ce0bd7c49e11cbde341756993a71618c5c0bf8e086de6caf56ad93e8588e1d94` | true | false |

## Safety

This lane only read the two authority archives. It did not modify root ZIPs, `others/`, score ledgers, queue manifests, or submissions.
