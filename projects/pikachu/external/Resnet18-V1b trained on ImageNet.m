(* ::Package:: *)

SetDirectory@NotebookDirectory[];


(* ::Input:: *)
(*<<DeepMath`*)


net = Import["Resnet18-V1b trained on ImageNet.WXF"]
NetTake[net, "Extractor"]


(*Anchors*(4+1+Classes)*)
new = NetChain[{
	"Extractor" -> NetExtract[net, "Extractor"],
	"Detector" -> ConvolutionLayer[2 * (2 + 1 + 4), {1, 1}]
}] // NetInitialize
NetSave[new, "Resnet18_V1"]



