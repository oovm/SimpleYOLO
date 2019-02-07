(* ::Package:: *)

SetDirectory@NotebookDirectory[];


(* ::Input:: *)
(*<<DeepMath`*)


net=Import["Resnet18-V1b trained on ImageNet.WXF"]
NetTake[net,"Extractor"]


NetSave[NetExtract[net,"Extractor"],"Resnet18_V1"]


NetSave[net,"Resnet18_V11111111"]
