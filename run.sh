CUDA_VISIBLE_DEVICES=0 python main.py
wait
echo "-----------------------bert-lstm -transformer pos konwledge done--------------------------"
cd /data/private/ldq/projects/beifen_parser/Parser_with_knowledge/
CUDA_VISIBLE_DEVICES=0 python main.py
wait
echo "-----------------------bert-lstm -transformer pos done--------------------------"