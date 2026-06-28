import numpy as np, onnx
from onnx import helper, TensorProto as TP, numpy_helper as nh

R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"

# Strategy (all uint8 grids [1,1,30,30] unless noted):
# G = green channel (slice ch3)  -> uint8
# nbr4 via Conv plus-kernel on G(float) -> count; T = (nbr==3)&G  (aitch seed)
# group-A corner seed, group-B corner seed (each from directional shifted masks)
# flood A,B,T (7 plus-dilations each, masked by G)
# you = floodA & floodB ; out: red if floodT; elif you -> magenta; else green->blue
# Build one-hot output [1,10,30,30].

DEPTH=7
nodes=[]; inits=[]
def C(name,arr): 
    inits.append(nh.from_array(arr.astype(arr.dtype),name)); return name

# input float [1,10,30,30]
# G = slice channel 3:4  -> [1,1,30,30] float
nodes.append(helper.make_node("Slice","Gf_in".split(),["Gf"],
    starts=None)) # placeholder, will set via inits below
# Use Gather to pick channel 3 (cheaper than slice w/ many inits)
# Actually use Slice with init tensors
inits=[]; nodes=[]
def add_init(name,arr): inits.append(nh.from_array(arr,name)); return name
add_init("ch3_start",np.array([3],np.int64))
add_init("ch3_end",np.array([4],np.int64))
add_init("ch1_axis",np.array([1],np.int64))
add_init("ch1_step",np.array([1],np.int64))
nodes.append(helper.make_node("Slice",["input","ch3_start","ch3_end","ch1_axis","ch1_step"],["Gf"]))  # [1,1,30,30] float
# G uint8
nodes.append(helper.make_node("Cast",["Gf"],["G"],to=TP.UINT8))

# directional neighbor masks (shift G). Use Pad+Slice or Conv. Use Conv with directional kernels on Gf.
# nbr4 count via Conv plus kernel
plus=np.array([[[[0,1,0],[1,0,1],[0,1,0]]]],np.float32)
add_init("Wplus",plus)
nodes.append(helper.make_node("Conv",["Gf","Wplus"],["nbrf"],pads=[1,1,1,1]))  # [1,1,30,30] float
# T seed = (nbr==3) & green
add_init("three",np.array([3.0],np.float32))
nodes.append(helper.make_node("Equal",["nbrf","three"],["nbr3"]))  # bool
nodes.append(helper.make_node("Cast",["nbr3"],["nbr3u"],to=TP.UINT8))
nodes.append(helper.make_node("Mul",["nbr3u","G"],["Tseed"]))  # uint8

# directional shifted green via Conv kernels (up/down/left/right presence)
def dirconv(name,kern):
    add_init("W"+name,np.array([[kern]],np.float32))
    nodes.append(helper.make_node("Conv",["Gf","W"+name],[name+"f"],pads=[1,1,1,1]))
    nodes.append(helper.make_node("Cast",[name+"f"],[name],to=TP.UINT8))
# kernel that picks the neighbor at offset: value at (dr,dc) -> place 1 there
# Conv flips kernel; to get neighbor in direction (dr,dc), kernel has 1 at (-dr,-dc) center-based
def k(dr,dc):
    a=np.zeros((3,3),np.float32); a[1-dr,1-dc]=1; return a
dirconv("UP",k(-1,0))    # UP = green pixel above exists
dirconv("DN",k(1,0))
dirconv("LF",k(0,-1))
dirconv("RT",k(0,1))
# corner group A = {SE bend: dn&rt, NW bend: up&lf}, masked green
nodes.append(helper.make_node("Mul",["DN","RT"],["dnrt"]))
nodes.append(helper.make_node("Mul",["UP","LF"],["uplf"]))
# group A presence (either), times green, and must be exactly-2-nbr corner (perp). 
# Use Max to OR
nodes.append(helper.make_node("Max",["dnrt","uplf"],["Araw"]))
nodes.append(helper.make_node("Mul",["Araw","G"],["Aseed"]))
# group B = {NE: up&rt, SW: dn&lf}
nodes.append(helper.make_node("Mul",["UP","RT"],["uprt"]))
nodes.append(helper.make_node("Mul",["DN","LF"],["dnlf"]))
nodes.append(helper.make_node("Max",["uprt","dnlf"],["Braw"]))
nodes.append(helper.make_node("Mul",["Braw","G"],["Bseed"]))

# Pack seeds into [1,3,30,30]: channels [Tseed,Aseed,Bseed]
nodes.append(helper.make_node("Concat",["Tseed","Aseed","Bseed"],["S0"],axis=1))  # [1,3,30,30] uint8
# G3 = green broadcast to 3 channels for masking
nodes.append(helper.make_node("Concat",["G","G","G"],["G3"],axis=1))
# flood: DEPTH times MaxPool(3) then Mul G3
cur="S0"
for i in range(DEPTH):
    mp=f"mp{i}"; fl=f"fl{i}"
    nodes.append(helper.make_node("MaxPool",[cur],[mp],kernel_shape=[3,3],pads=[1,1,1,1],strides=[1,1]))
    nodes.append(helper.make_node("Mul",[mp,"G3"],[fl]))
    cur=fl
# split channels
add_init("c0s",np.array([0],np.int64)); add_init("c1s",np.array([1],np.int64))
add_init("c2s",np.array([2],np.int64)); add_init("c3s",np.array([3],np.int64))
nodes.append(helper.make_node("Slice",[cur,"c0s","c1s","ch1_axis","ch1_step"],["fT"]))
nodes.append(helper.make_node("Slice",[cur,"c1s","c2s","ch1_axis","ch1_step"],["fA"]))
nodes.append(helper.make_node("Slice",[cur,"c2s","c3s","ch1_axis","ch1_step"],["fB"]))
# you = fA & fB ; aitch = fT
nodes.append(helper.make_node("Mul",["fA","fB"],["youM"]))
# Now build output one-hot:
# red(ch2) = fT ; magenta(ch6)= youM & not fT & G ; blue(ch1)= G & not fT & not youM
# not fT
add_init("one_u",np.array([1],np.uint8))
nodes.append(helper.make_node("Sub",["one_u","fT"],["notT"]))
nodes.append(helper.make_node("Mul",["youM","notT"],["mag"]))     # magenta where you & not aitch
# blue = G & notT & not mag
nodes.append(helper.make_node("Sub",["one_u","mag"],["notMag"]))
nodes.append(helper.make_node("Mul",["G","notT"],["GnotT"]))
nodes.append(helper.make_node("Mul",["GnotT","notMag"],["blue"]))
# red = fT (already green by construction, but ensure & G)
nodes.append(helper.make_node("Mul",["fT","G"],["red"]))
# zero channel
add_init("zero_u",np.zeros((1,1,30,30),np.uint8))
# output channels: ch0=zero, ch1=blue, ch2=red, 3,4,5=zero, ch6=mag, 7,8,9=zero
order=["zero_u","blue","red","zero_u","zero_u","zero_u","mag","zero_u","zero_u","zero_u"]
nodes.append(helper.make_node("Concat",order,["out_u"],axis=1))  # [1,10,30,30] uint8
nodes.append(helper.make_node("Cast",["out_u"],["output"],to=TP.FLOAT))

g=helper.make_graph(nodes,"t364",
    [helper.make_tensor_value_info("input",TP.FLOAT,[1,10,30,30])],
    [helper.make_tensor_value_info("output",TP.FLOAT,[1,10,30,30])],
    inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",13)])
m.ir_version=9
onnx.save(m, R+"/scripts/golf/scratch_wave/task364_cand.onnx")
print("saved")
