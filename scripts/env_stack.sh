export MODEL_REPOSITORY=/home/amine/Desktop/examples/model_repository
export TRITON="/home/amine/model_serving"
export PATH=$PATH:$TRITON
echo $PATH
alias triton='source $TRITON/triton_server'