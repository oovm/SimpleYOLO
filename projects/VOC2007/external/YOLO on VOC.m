(* ::Package:: *)

SetDirectory@NotebookDirectory[];


(* ::Input:: *)
(*<<DeepMath`*)


net=Import@"YOLO on VOC-model.WXF"
NetSave[net,"YOLO on VOC"]



