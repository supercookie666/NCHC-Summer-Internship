### 參數設定區 ###
## 工作目錄
WORKDIR=/home/chander92811/yolov9 ## 更換為你的yolov9 目錄
cd $WORKDIR

## SLURM 環境
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
if [ -z "$MASTER_ADDR" ]; then
    echo "oh! why MASTER_ADDR not found!"
    MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
fi

#NGPU=$SLURM_GPUS_ON_NODE #這個值常抓不到
#NGPU=$NPROC_PER_NODE # NPROC_PER_NODE是gpu數但在這邊也抓錯
if [ -z "$NGPU" ]; then
    echo "oh! why NPROC_PER_NODE not found!"
    NGPU=$(nvidia-smi -L | wc -l)  # 等於 $SLURM_GPUS_ON_NODE
fi

MASTER_PORT=9527
DEVICE_LIST=$(seq -s, 0 $(($NGPU-1)) | paste -sd, -) # 0,1,...n-1
NNODES=${SLURM_NNODES:-1}               # 節點總數，默認為 1
NODE_RANK=${SLURM_NODEID}            # 當前節點的 rank，默認為 0

echo "Debug Information:"
echo "==================="
echo "SLURM_NODEID: $NODE_RANK"
echo "SLURM_NNODES: $NNODES"
echo "SLURM_GPUS_ON_NODE: $NGPU"
echo "Device: $DEVICE_LIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Current Hostname: $(hostname)"
echo "==================="

### 環境檢查區 ### 
## Debug: 確認 Python 路徑與版本
echo "Python Path and Version:"
echo "==================="
which python
python --version
echo "PYTHONPATH: $PYTHONPATH"
echo "==================="

echo "Activated Conda Environment:"
echo "==================="
python -c "import sys; print('\n'.join(sys.path))"
wandb login b0873e135aede1107b9524e83fbd2d526ac60861  #好像只有初次需設定
python -c 'import wandb'
python -c 'import torch; print(torch.__version__)'
echo "==================="
echo "env.py"
python env.py
echo "==================="

### 執行訓練命令 ###
## 超參數設定
NBatch=32       # v100 超過 254會failed ，4的倍數調整
NEpoch=150      # 約 20 mins / per Epoch
NWorker=8       # cpu = gpu x 4, worker < cpu (是看單節點內持有之cpu)

## 訓練 train_dual.py 命令 (動態設置 nproc_per_node 和 nnodes)
TRAIN_CMD="torchrun --nproc_per_node=$NGPU --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
          train_dual.py \
          --workers $NWorker \
	      --device $DEVICE_LIST \
	      --batch $NBatch \
          --data data/kitti_central.yaml \
	      --img 640 \
	      --cfg models/detect/yolov9-c.yaml \
	      --project FL-GPUs-Segmentation \
	      --entity qqhair000-national-chung-cheng-university \
          --weights '' \
	      --name fed-central \
	      --hyp data/hyps/hyp.scratch-high.yaml \
	      --min-item 0 \
          --epochs $NEpoch \
	      --close-mosaic 25 \
          --sync-bn"

## 印出完整的訓練命令
echo "Executing Training Command:"
echo "$TRAIN_CMD"
echo "==================="
$TRAIN_CMD

## 檢查執行結果
if [ $? -ne 0 ]; then
  echo "Error: TRAIN_CMD execution failed on node $(hostname)" >&2
  exit 1
fi