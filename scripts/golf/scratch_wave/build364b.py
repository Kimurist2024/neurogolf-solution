import numpy as np, onnx
from onnx import helper, TensorProto as TP, numpy_helper as nh
R="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
DEPTH=7
nodes=[]; inits=[]
def ai(name,arr): inits.append(nh.from_array(arr,name)); return name
ai("ch3_s",np.array([3],np.int64)); ai("ch3_e",np.array([4],np.int64))
ai("ax1",np.array([1],np.int64)); ai("st1",np.array([1],np.int64))
nodes.append(helper.make_node("Slice",["input","ch3_s","ch3_e","ax1","st1"],["Gf"]))  # [1,1,30,30] f32
nodes.append(helper.make_node("Cast",["Gf"],["Gh"],to=TP.FLOAT16))
# nbr4 conv
ai("Wplus",np.array([[[[0,1,0],[1,0,1],[0,1,0]]]],np.float32))
nodes.append(helper.make_node("Conv",["Gf","Wplus"],["nbrf"],pads=[1,1,1,1]))
ai("three",np.array([3.0],np.float32))
nodes.append(helper.make_node("Equal",["nbrf","three"],["nbr3"]))
nodes.append(helper.make_node("Cast",["nbr3"],["nbr3h"],to=TP.FLOAT16))
nodes.append(helper.make_node("Min",["nbr3h","Gh"],["Tseed"]))
def dk(name,dr,dc):
    a=np.zeros((3,3),np.float32); a[1-dr,1-dc]=1
    ai("W"+name,np.array([[a]],np.float32))
    nodes.append(helper.make_node("Conv",["Gf","W"+name],[name+"f"],pads=[1,1,1,1]))
    nodes.append(helper.make_node("Cast",[name+"f"],[name],to=TP.FLOAT16))
dk("UP",-1,0); dk("DN",1,0); dk("LF",0,-1); dk("RT",0,1)
nodes.append(helper.make_node("Min",["DN","RT"],["dnrt"]))
nodes.append(helper.make_node("Min",["UP","LF"],["uplf"]))
nodes.append(helper.make_node("Max",["dnrt","uplf"],["Araw"]))
nodes.append(helper.make_node("Min",["Araw","Gh"],["Aseed"]))
nodes.append(helper.make_node("Min",["UP","RT"],["uprt"]))
nodes.append(helper.make_node("Min",["DN","LF"],["dnlf"]))
nodes.append(helper.make_node("Max",["uprt","dnlf"],["Braw"]))
nodes.append(helper.make_node("Min",["Braw","Gh"],["Bseed"]))
nodes.append(helper.make_node("Concat",["Tseed","Aseed","Bseed"],["S0"],axis=1))
nodes.append(helper.make_node("Concat",["Gh","Gh","Gh"],["G3"],axis=1))
cur="S0"
for i in range(DEPTH):
    nodes.append(helper.make_node("MaxPool",[cur],[f"mp{i}"],kernel_shape=[3,3],pads=[1,1,1,1],strides=[1,1]))
    nodes.append(helper.make_node("Min",[f"mp{i}","G3"],[f"fl{i}"]))
    cur=f"fl{i}"
ai("c0",np.array([0],np.int64)); ai("c1",np.array([1],np.int64)); ai("c2",np.array([2],np.int64)); ai("c3",np.array([3],np.int64))
nodes.append(helper.make_node("Slice",[cur,"c0","c1","ax1","st1"],["fT"]))
nodes.append(helper.make_node("Slice",[cur,"c1","c2","ax1","st1"],["fA"]))
nodes.append(helper.make_node("Slice",[cur,"c2","c3","ax1","st1"],["fB"]))
nodes.append(helper.make_node("Min",["fA","fB"],["youM"]))
ai("oneh",np.array([1.0],np.float16))
nodes.append(helper.make_node("Sub",["oneh","fT"],["notT"]))
nodes.append(helper.make_node("Min",["youM","notT"],["mag0"]))
nodes.append(helper.make_node("Min",["mag0","Gh"],["mag"]))
nodes.append(helper.make_node("Sub",["oneh","mag"],["notMag"]))
nodes.append(helper.make_node("Min",["Gh","notT"],["GnT"]))
nodes.append(helper.make_node("Min",["GnT","notMag"],["blue"]))
nodes.append(helper.make_node("Min",["fT","Gh"],["red"]))
ai("zeroh",np.zeros((1,1,30,30),np.float16))
order=["zeroh","blue","red","zeroh","zeroh","zeroh","mag","zeroh","zeroh","zeroh"]
nodes.append(helper.make_node("Concat",order,["out_h"],axis=1))
nodes.append(helper.make_node("Cast",["out_h"],["output"],to=TP.FLOAT))
g=helper.make_graph(nodes,"t364b",
  [helper.make_tensor_value_info("input",TP.FLOAT,[1,10,30,30])],
  [helper.make_tensor_value_info("output",TP.FLOAT,[1,10,30,30])], inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",13)]); m.ir_version=9
onnx.save(m,R+"/scripts/golf/scratch_wave/task364_b.onnx"); print("saved")
