### preprocess script ###
. setting.sh

mkdir -p ${PROCESS_DIR}
TEXT=work

fairseq-preprocess \
    --source-lang ja --target-lang en \
    --trainpref ${data}/kyoto-train.tok \
    --validpref ${data}/kyoto-dev.tok \
    --testpref ${TEXT}/kyoto-test.tok \
    --destdir ${PROCESS_DIR}
    --workers 20
