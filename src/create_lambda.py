import boto3
import argparse

def create_lammbda(role ,
                   bucket_name ,
                   layer_arn ,
                   DB_NAME ,
                   DB_USER ,
                   CLUSTER_IDENTIFIER ,
                   PIPELINE_NAME,
                   region_name):

    lambda_client = boto3.client('lambda', region_name = region_name)
    try :
        lambda_client.create_function(FunctionName='myfunction-lambda',
            Runtime='python3.8',
            Role=role,
            Handler='lambda_function.lambda_handler',
            Code ={
                'S3Bucket': bucket_name,
                'S3Key': 'lambda-code/my-deployment-package.zip',
            },
            Layers=[
                layer_arn,
            ],
    Environment={'Variables': {'DB_NAME': DB_NAME,
                               'DB_USER': DB_USER,
                               'CLUSTER_IDENTIFIER': CLUSTER_IDENTIFIER,
                               'ROLE_ARN': role,
                               'PIPELINE_NAME': PIPELINE_NAME,
                               'BUCKET_NAME': bucket_name,
                               'REGION': region_name,
                               }
                                          },
            Description='lambda',
            Timeout=900,
            MemorySize=128,
            Publish=True )
    except Exception as e:
        print (e)
        print('Error Creating a lambda')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='default')
    parser.add_argument('--db_user', type=str, default='default')
    parser.add_argument('--cluster_id', type=str, default='default')
    parser.add_argument('--pipeline_name', type=str, default='default')
    parser.add_argument('--layer_arn', type=str, default='default')
    parser.add_argument('--role_arn', type=str, default='default')
    parser.add_argument('--bucket_name', type=str, default='default')
    parser.add_argument('--region_name', type=str, default='us-east-1')
    
    args, _ = parser.parse_known_args()
    db_name = args.db_name
    db_user = args.db_user
    cluster_id = args.cluster_id
    role_arn = args.role_arn
    layer_arn = args.layer_arn
    bucket_name = args.bucket_name
    pipeline_name = args.pipeline_name
    region_name = args.region_name

    print(db_name , db_user ,cluster_id ,role_arn, layer_arn ,bucket_name ,pipeline_name )
    create_lammbda(role_arn ,bucket_name , layer_arn , db_name , db_user , cluster_id , pipeline_name, region_name)
    
