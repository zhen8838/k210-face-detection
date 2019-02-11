CKPT:=none
INNODE:=input
OUTNODE:=predict
TFLITE:=mobilenet_v1_0.5_224_320_frozen.tflite
PB:=Freeze_save.pb.pb

train:
	python3 train.py --pre_ckpt log/20190207-213702

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
                    --image_h 224 --image_w 320 \
                    --dataset_pic_path "dataset/flowers" \
                    --model_loader "model_loader/pb" \
                    --pb_path "pb_files/${PB}" \
                    --tensor_output_name ${OUTNODE}