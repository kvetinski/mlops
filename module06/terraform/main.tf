# LOCALSTACK ONLY. Dummy credentials and a loopback endpoint keep this
# exercise separate from any real AWS account.
provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_region_validation      = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true

  endpoints {
    s3 = "http://127.0.0.1:4566"
  }
}

resource "aws_s3_bucket" "batch_predictions" {
  bucket        = var.bucket_name
  force_destroy = true # safe only for this disposable LocalStack bucket

  tags = {
    Project     = "mlops-zoomcamp"
    Module      = "06"
    Environment = "localstack"
  }
}

output "bucket_name" {
  value       = aws_s3_bucket.batch_predictions.bucket
  description = "Bucket created in LocalStack S3"
}
