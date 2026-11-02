variable "bucket_name" {
  type        = string
  description = "Disposable local S3 bucket for batch integration tests"
  default     = "nyc-duration-iac"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", var.bucket_name))
    error_message = "Use a 3-63 character lowercase S3 bucket name."
  }
}
