CKPT:=""
INNODE:=Input_image
OUTNODE:=Yolo/Final/conv2d/BiasAdd
TFLITE:=mobilenet_v1_0.5_224_320_frozen.tflite
PB:=Freeze_save.pb.pb
H=240
W=320
IAA=False
ILR=0.0005
MAXEP=10

train_pureconv:
	python3 train.py --pre_ckpt ${CKPT} --image_size 240 320  --model_def pureconv --augmenter ${IAA} --init_learning_rate ${ILR} --max_nrof_epochs ${MAXEP}
	
train_yoloconv:
	python3 train.py --image_size 224 320 --model_def yoloconv --augmenter True

freeze:
	python3 freeze_graph.py \
			 ${CKPT} \
			 ${PB} \
			 ${OUTNODE}  
tflite:
	toco --graph_def_file=${PB} \
	--output_file=${TFLITE} \
	--output_format=TFLITE \
	--input_shape=1,224,320,3 \
	--input_array=${INNODE} \
	--output_array=${OUTNODE} \
	--inference_type=FLOAT && \
	cp -f ${TFLITE} ~/Documents/nncase/tflites/

nncase_convert:
	cd ~/Documents/nncase/ && \
	/home/zqh/Documents/nncase/src/NnCase.Cli/bin/Debug/netcoreapp3.0/ncc \
					-i tflite -o k210code \
					--dataset dataset/flowers \
					--postprocess n1to1 \
					tflites/${TFLITE} build/model.c

kmodel_convert:
	cp -f ${PB} ~/Documents/kendryte-model-compiler/pb_files/ && \
	cd ~/Documents/kendryte-model-compiler/ && \
	python3 __main__.py --dataset_input_name ${INNODE}:0 \
                    --dataset_loader "dataset_loader/img_neg1_1.py" \
                    --image_h 240 --image_w 320 \
                    --dataset_pic_path "dataset/flowers" \
                    --model_loader "model_loader/pb" \
                    --pb_path "pb_files/${PB}" \
                    --tensor_output_name ${OUTNODE}