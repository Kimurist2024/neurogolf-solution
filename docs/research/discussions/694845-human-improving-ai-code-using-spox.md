# human improving Ai code : using Spox

- Topic ID: 694845
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/694845
- Author: hengck23 (@hengck23)
- Posted: 2026-04-27T11:56:00.738502300Z
- Votes: 7
- Total messages: 1

## Body

i want to introduce the framework:
https://github.com/Quantco/spox/

this is easier to use than the onnx helper. so i ask my agent to code in Spox, then i can modify it. If you still want to use  onnx helper, you can ask agent to translate to it at the final tweak. My solution may not be the best, love to know what other kagglers are doing.

advantage:
- easy to chnage datatype float, uint8, bool
- easy to swtich opset
- easy to show debug ouput
- you make modular code and shared sub functions
- easy to swap between methods, e.g. flatten vs reshape, negative pad vs pad+concat, tile vs expand ...
- you can monitor score step by step

as an example, i show task008 (detecting and moving object) which i can get 15.110 by coding from scratch with agent help. 

```
def make_task008():

    #not allowed to change this
    input = argument(Tensor(np.float32, (1, 10, 30, 30))) 

    #x_red = input16[:, 2:3]  # 1x1x16x16
    x_red = op.slice(
        input,
        starts=op.constant(value=np.array([2, 0, 0], dtype=np.int64)),
        ends=op.constant(value=np.array([3, 16, 16], dtype=np.int64)),
        axes=op.constant(value=np.array([1, 2, 3], dtype=np.int64)),
        steps=op.constant(value=np.array([1, 1, 1], dtype=np.int64)),
    )
    x_blue = op.slice(
        input,
        starts=op.constant(value=np.array([8, 0, 0], dtype=np.int64)),
        ends=op.constant(value=np.array([9, 16, 16], dtype=np.int64)),
        axes=op.constant(value=np.array([1, 2, 3], dtype=np.int64)),
        steps=op.constant(value=np.array([1, 1, 1], dtype=np.int64)),
    )
    x_bg = op.slice(
        input,
        starts=op.constant(value=np.array([0, 0, 0], dtype=np.int64)),
        ends=op.constant(value=np.array([1, 16, 16], dtype=np.int64)),
        axes=op.constant(value=np.array([1, 2, 3], dtype=np.int64)),
        steps=op.constant(value=np.array([1, 1, 1], dtype=np.int64)),
    )

    #occpancy = x[:, 0:1] + x[:, 8:9] + x[:, 2:3]
    data_type = np.uint8
    x_bg   = op.cast(x_bg, to=np.uint8)
    x_red  = op.cast(x_red, to=np.uint8)
    x_blue = op.cast(x_blue, to=np.uint8)
    occpancy = op.add(op.add(x_bg, x_blue), x_red)

    #find bounding box
    idx16 = op.constant(value=np.arange(1, 17, dtype=np.uint8))
    v17 = op.constant(value=np.array(17, dtype=np.uint8))
    v0 = op.constant(value=np.array(0, dtype=np.uint8))
    v1 = op.constant(value=np.array(1, dtype=np.uint8))

    xmask = op.reduce_max(x_red, axes=(2,))#1x1x16
    ymask = op.reduce_max(x_red, axes=(3,))#1x1x16
    xmask = op.flatten(xmask)
    ymask = op.flatten(ymask) #flatten wins reshape

    x_has = op.cast(xmask, to=np.bool_)
    y_has = op.cast(ymask, to=np.bool_)

    #index_16 = torch.arange(1,17).to(torch.int8)
    x_idx = op.mul(idx16, xmask)
    y_idx = op.mul(idx16, ymask)

    # for min: replace false positions with 17
    x_for_min = op.where(x_has, x_idx, v17)
    y_for_min = op.where(y_has, y_idx, v17) ###cannot for int8

    # x1/x2/y1/y2 are scalar int8
    x1 = op.sub(op.reduce_min(x_for_min, keepdims=0), v1)
    x2 = op.sub(op.reduce_max(x_idx, keepdims=0), v1)
    y1 = op.sub(op.reduce_min(y_for_min, keepdims=0), v1)
    y2 = op.sub(op.reduce_max(y_idx, keepdims=0), v1)

    # score up to this point:
    #It appears to require 608 MACs + 1552 bytes + 28 params, yielding cost = 2188 and score = 17.309

    #find box for blue
    #blue is box 2x2
    if 1:
        x_blue_flat = op.flatten(x_blue)
        bxy1 = op.arg_max(
            x_blue_flat,
            axis=1,
            keepdims=0,
            select_last_index=0,
        )#first non-zero

        four = op.constant(value=np.array(4, dtype=np.uint8))
        sixteen = op.constant(value=np.array(16, dtype=np.uint8))
        #mask15 = op.constant(value=np.array(15, dtype=np.uint8))

        bxy1 = op.cast(bxy1, to=np.uint8)
        #by1 = op.bit_shift(bxy1, four, direction="RIGHT")
        by1 = op.div(bxy1, sixteen)   # y = idx // 16
        #bx1 = op.bitwise_and(bxy1, mask15)
        bx1 = op.mod(bxy1, sixteen)
        #by1 = bxy1//16
        #bx1 = bxy1%16

    # score up to this point:
    # "debug": op.cast(bx1, to=np.uint8),
    #It appears to require 609 MACs + 3146 bytes + 34 params, yielding cost = 3789 and score = 16.760

    #---
    # print all distance between red and blue
    # top   =by1-y2-1
    # bot   =y1-(by1+2)
    # left  =bx1-x2-1
    # right =x1-(bx1+2)  #ask chtgpt to tasnslate

    by1 = op.cast(by1, to=np.int8)
    bx1 = op.cast(bx1, to=np.int8)
    y1 = op.cast(y1, to=np.int8)
    y2 = op.cast(y2, to=np.int8)
    x1 = op.cast(x1, to=np.int8)
    x2 = op.cast(x2, to=np.int8)

    one = op.constant(value=np.array(1, dtype=np.int8))
    two = op.constant(value=np.array(2, dtype=np.int8))
    zero = op.constant(value=np.array(0, dtype=np.int8))
    maxv = op.constant(value=np.array(127, dtype=np.int8))  # upper bound
    # top = by1 - y2 - 1
    top = op.sub(op.sub(by1, y2), one)
    # bot = y1 - (by1 + 2)
    bot = op.sub(y1, op.add(by1, two))
    # left = bx1 - x2 - 1
    left = op.sub(op.sub(bx1, x2), one)
    # right = x1 - (bx1 + 2)
    right = op.sub(x1, op.add(bx1, two))

    #choose non zero to move red towards blue
    top_pos = op.clip(top, zero, maxv)
    bot_pos = op.clip(bot, zero, maxv)
    left_pos = op.clip(left, zero, maxv)
    right_pos = op.clip(right, zero, maxv)

    # vertical move:
    # top positive  => move down  +top
    # bot positive  => move up    -bot
    dy = op.sub(top_pos, bot_pos)

    # horizontal move:
    # left positive  => move right +left
    # right positive => move left  -right
    dx = op.sub(left_pos, right_pos)

    # score up to this point:
    # "debug": op.cast(dx, to=np.uint8),
    # It appears to require 618 MACs + 3155 bytes + 34 params, yielding cost = 3807 and score = 16.755

    #---
    dy64 = op.cast(dy, to=np.int64)
    dx64 = op.cast(dx, to=np.int64)

    zero64 = op.constant(value=np.array(0, dtype=np.int64))
    shape1 = op.constant(value=np.array([1], dtype=np.int64))

    def as1(v):
        return op.reshape(v, shape1)

    pads = op.concat(
        [
            as1(zero64),  # N begin
            as1(zero64),  # C begin
            as1(dy64),  # H begin
            as1(dx64),  # W begin
            as1(zero64),  # N end
            as1(zero64),  # C end
            as1(op.neg(dy64)),  # H end
            as1(op.neg(dx64)),  # W end
        ],
        axis=0,
    )

    zero_u8 = op.constant(value=np.array(0, dtype=np.uint8))

    red_shifted = op.pad(
        x_red,
        pads,
        constant_value=zero_u8,
        mode="constant",
    )



    ################################################################3
    #output = np.zeros(1,10,30,30)#np.uint8
    #output[0,2]=red_shifted
    #output[0,8]=x_blue
    #output[0,0]=occpancy-red_shifted-x_blue
    #output = x_blue #(1, 1, 16, 16)
    #output = red_shifted #(1, 1, 16, 14)

    bg = op.sub(op.sub(occpancy, red_shifted), x_blue)
    zero = op.constant(value=np.array(0, dtype=np.uint8))
    zero_ch = op.expand(
        zero,
        op.constant(value=np.array([1, 1, 16, 16], dtype=np.int64))
    )
    channels = [
        bg,  # 0
        zero_ch,  # 1
        red_shifted,  # 2
        zero_ch,  # 3
        zero_ch,  # 4
        zero_ch,  # 5
        zero_ch,  # 6
        zero_ch,  # 7
        x_blue,  # 8
        zero_ch,  # 9
    ]
    output = op.concat(channels, axis=1)
    zero64 = op.constant(value=np.array(0, dtype=np.int64))

    pads = op.constant(
        value=np.array([
            0, 0,  # N, C begin
            0, 0,  # H, W begin
            0, 0,  # N, C end
            14, 14  # H, W end
        ], dtype=np.int64)
    )

    zero_u8 = op.constant(value=np.array(0, dtype=np.uint8))
    output = op.pad(
        output,  # your 1x10x16x16
        pads,
        constant_value=zero_u8,
        mode="constant",
    )


    ###############################3
    model = spox.build(
        inputs={"input": input},
        outputs={
            "output": output,
            "y2": y2,
            "y1": y1,
            "x2": x2,
            "x1": x1,
            "dx": dx,
            "dy": dy,
            "debug": op.cast(red_shifted, to=np.uint8),
        },
    )
    onnx.checker.check_model(model)
   #  model, ok = simplify(model) #squeeze a few more point
    return model

```
```

Performance stats:
Name         Type         Forward_MACs  FPercent      Memory  MPercent      Params  PPercent    InShape    OutShape
-----------  ---------  --------------  ----------  --------  ----------  --------  ----------  ---------  ----------
ReduceMax_2  ReduceMax             256  15.45%            16  1.09%              0  0.00%       1x1x16x16  1x1x1x16
Mul_0        Mul                    16  0.97%             32  2.18%             16  100.00%     16         1x16
Sub_0        Sub                     0  0.00%              0  0.00%              0  0.00%       0          0
Sub_10       Sub                     1  0.06%              1  0.07%              0  0.00%       0          1
Sub_13       Sub                   256  15.45%           256  17.40%             0  0.00%       1x1x16x16  1x1x16x16
Sub_2        Sub                     1  0.06%              1  0.07%              0  0.00%       1          1
Sub_7        Sub                     1  0.06%              1  0.07%              0  0.00%       1          1
Where_0      Where                   0  0.00%             16  1.09%              0  0.00%       1x16       1x16
ReduceMin_1  ReduceMin              16  0.97%              0  0.00%              0  0.00%       1x16       0
Div_0        Div                     4  0.24%              8  0.54%              0  0.00%       1          1
Neg_0        Neg                     1  0.06%              8  0.54%              0  0.00%       1          1
Mul_1        Mul                    16  0.97%             16  1.09%              0  0.00%       16         1x16
Sub_11       Sub                     1  0.06%              1  0.07%              0  0.00%       1          1
Sub_3        Sub                     0  0.00%              0  0.00%              0  0.00%       0          0
Flatten_1    Flatten                 0  0.00%             16  1.09%              0  0.00%       1x1x16x1   1x16
Flatten_2    Flatten                 0  0.00%             16  1.09%              0  0.00%       1x1x1x16   1x16
Sub_5        Sub                     1  0.06%              1  0.07%              0  0.00%       1          1
Clip_1       Clip                    2  0.12%              1  0.07%              0  0.00%       1          1
Sub_8        Sub                     1  0.06%              1  0.07%              0  0.00%       1          1
Sub_1        Sub                     1  0.06%              1  0.07%              0  0.00%       1          1
ReduceMax_0  ReduceMax             256  15.45%            16  1.09%              0  0.00%       1x1x16x16  1x1x16x1
Clip_0       Clip                    2  0.12%              1  0.07%              0  0.00%       1          1
Add_1        Add                   256  15.45%           256  17.40%             0  0.00%       1x1x16x16  1x1x16x16
Sub_12       Sub                   256  15.45%           256  17.40%             0  0.00%       1x1x16x16  1x1x16x16
Sub_4        Sub                     1  0.06%              1  0.07%              0  0.00%       0          1
ReduceMax_1  ReduceMax              16  0.97%              0  0.00%              0  0.00%       1x16       0
Neg_1        Neg                     1  0.06%              8  0.54%              0  0.00%       1          1
Mod_0        Mod                     1  0.06%              1  0.07%              0  0.00%       1          1
Clip_2       Clip                    2  0.12%              1  0.07%              0  0.00%       1          1
ReduceMin_0  ReduceMin              16  0.97%              0  0.00%              0  0.00%       1x16       0
Add_0        Add                   256  15.45%           256  17.40%             0  0.00%       1x1x16x16  1x1x16x16
ReduceMax_3  ReduceMax              16  0.97%              0  0.00%              0  0.00%       1x16       0
Sub_6        Sub                     0  0.00%              0  0.00%              0  0.00%       0          0
Where_1      Where                   0  0.00%             16  1.09%              0  0.00%       1x16       1x16
Add_2        Add                     1  0.06%              1  0.07%              0  0.00%       1          1
ArgMax_0     ArgMax                  0  0.00%              8  0.54%              0  0.00%       1x256      1
Sub_9        Sub                     0  0.00%              0  0.00%              0  0.00%       0          0
Clip_3       Clip                    2  0.12%              1  0.07%              0  0.00%       1          1
Add_3        Add                     1  0.06%              1  0.07%              0  0.00%       1          1
Flatten_0    Flatten                 0  0.00%            256  17.40%             0  0.00%       1x1x16x16  1x256
Total        _                   1,657  100%           1,471  100%              16  100%        _          _

It appears to require 1657 MACs + 17762 bytes + 305 params, yielding cost = 19724 and score = 15.110



```

## Comments (1)

- **hengck23** (2026-04-27T15:57:49.317Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  i finally can undersatnce what he is writing
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F8fcb8f74a2cfeda41ec0a1bc95f0e6cc%2FSelection_3170.png?generation=1777305412960980&alt=media)
  
  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ffb728c5c27d09dc395b7168ac7eea002%2FSelection_3171.png?generation=1777305466107816&alt=media)
