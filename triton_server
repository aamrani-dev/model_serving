if [ "$1" == "start" ]; then
	sudo killall tritonserver

	echo "Starting..."    
	model_repository=$MODEL_REPOSITORY
	n_gpus=1


	while getopts g: flag
	do
	    case "${flag}" in
	        g) n_gpus=${OPTARG};;
	    esac
	done

	sudo nvidia-docker run --gpus=$n_gpus --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$MODEL_REPOSITORY:/models nvcr.io/nvidia/tritonserver:21.04-py3 tritonserver --model-repository=/models --strict-model-config=false & 

	echo "...Started"

	export alive=1
else
	if [ "$1" == "stop" ]; then
		sudo killall tritonserver
		export alive=0
	else
		if [ "$1" == "is_alive" ]; then
			if [ "$alive" == 1 ]; then
				echo "THE SERVER IS RUNNING"
			else
				echo "THE SERVER IS NOT RUNNING"

			fi
		fi
	fi
fi