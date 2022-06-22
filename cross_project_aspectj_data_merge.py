#220212
#测试aspectj 将数据混合后训练
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/test.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/valid.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/codebase.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase_MIX.jsonl

***jdt
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/test.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/valid.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/codebase.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase_MIX.jsonl
*****

***birt
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/test.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/valid.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/codebase.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase_MIX.jsonl
*****

***swt
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/test.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/valid.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/codebase.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase_MIX.jsonl
*****

*** eclipse_platform_ui
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/test.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/valid.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/codebase.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase_MIX.jsonl
*****

********
tomcat

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/test.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/valid.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid_MIX.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/siamese_data/codebase.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl > /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase_MIX.jsonl
# 6个项目，每五个作为训练集，另一个作为测试集,把train.jsonl valid.jsonl test.jsonl codebase.jsonl 合并
# train : Eclipse UI ,JDT,BIRT,SWT,Tomcat 
# test: AspectJ
aspectj*************************************************
aspectj  birt  eclipse_platform_ui  jdt    swt   tomcat 

 codebase.jsonl 
 test_1.jsonl 
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl  >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid.jsonl>  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase_cross.jsonl

tomcat*************************************************
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid.jsonl>  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase_cross.jsonl

birt***************************************
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid.jsonl>  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase_cross.jsonl

eclipse_platform_ui ***************************************
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid.jsonl>  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase_cross.jsonl

jdt ***************************************
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid.jsonl>  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase_cross.jsonl


swt ***************************************
cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/train.jsonl  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/train.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/train.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/train_cross.jsonl


cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/valid.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/valid.jsonl>  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/valid_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/test.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/test.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/test_cross.jsonl

cat /data/hdj/data/CodeBERT/Siamese-model/dataset/java/birt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/eclipse_platform_ui/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/jdt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/tomcat/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl /data/hdj/data/CodeBERT/Siamese-model/dataset/java/aspectj/codebase.jsonl >  /data/hdj/data/CodeBERT/Siamese-model/dataset/java/swt/codebase_cross.jsonl
