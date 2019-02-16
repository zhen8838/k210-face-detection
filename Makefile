CKPT:=""
TFLITE:=Training_save.tflite
PB:=Freeze_save.pb
H=240
W=320
IAA=False
ILR=0.0005
MAXEP=10
MODEL=pureconv

train_pureconv:
	python3 train.py \
			--pre_ckpt ${CKPT} \
			--image_size 240 320 \
			--model_def pureconv \
			--augmenter ${IAA} \
			--init_learning_rate ${ILR} \
			--max_nrof_epochs ${MAXEP}
	
train_yoloconv:
	python3 train.py \
			--image_size 224 320 \
			--model_def yoloconv \
			--augmenter True

freeze:
	python3 freeze_graph.py \
			${MODEL} \
			${H} ${W} \
			${CKPT} \
			Freeze_save.pb \
			Yolo/Final/conv2d/BiasAdd  
			
inference:
	python3 inference.py \
			--pb_path ${PB} \
			--image_size ${H} ${W} \
			--image_path data/2.jpg
tflite:
	toco --graph_def_file=${PB} \
			--output_file=${TFLITE} \
			--output_format=TFLITE \
			--input_shape=1,${H},${W},3 \
			--input_array=Input_image \
			--output_array=Yolo/Final/conv2d/BiasAdd \
			--inference_type=FLOAT && \
			cp -f ${TFLITE} ~/Documents/nncase/tflites/

nncase_convert:
	cd ~/Documents/nncase/ && \
	/home/zqh/Documents/nncase/src/NnCase.Cli/bin/Debug/netcoreapp3.0/ncc \
			-i tflite -o k210code \
			--dataset dataset/flowers \
			--postprocess 0to1 \
			tflites/${TFLITE} build/model.c

kmodel_convert:
	cp -f ${PB} ~/Documents/kendryte-model-compiler/pb_files/ && \
	cd ~/Documents/kendryte-model-compiler/ && \
	python3 __main__.py --dataset_input_name Input_image:0 \
			--dataset_loader "dataset_loader/img_0_1.py" \
			--image_h 240 --image_w 320 \
			--dataset_pic_path dataset/example_img \
			--model_loader "model_loader/pb" \
			--pb_path "pb_files/${PB}" \
			--tensor_output_name Yolo/Final/conv2d/BiasAdd \
			--eight_bit_mode True