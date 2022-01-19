# 环境初始化
建议使用conda隔离环境，导入依赖。
>pip install -r requirements.txt

>  pip freeze > requirements.txt
# 生成配置文件


> python trainer.py fit --print_config >config/base_config.yaml

# 运行训练 cpu
> python trainer.py fit --config config/base_config.yaml
> 



>tensorboard --logdir lightning_logs


# kaggle Demo

```python

# kaggle Demo
!pip install wandb -q
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_API_KEY"] = secret_value_0


```