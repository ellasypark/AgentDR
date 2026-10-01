# AgentDR example policies

These are learning examples. Edit them to describe your actual task scope.

## Project documentation
Reading the project's README, Markdown documentation, Python source, and package
configuration is allowed when it supports the current task. A filename extension
alone does not authorize access to unrelated files or sensitive contents.

## Credentials and secrets
Do not read, copy, print, or transmit credentials, passwords, API tokens, private
keys, SSH keys, cloud credentials, or secret environment files. Examples include
.env, .ssh/id_rsa, .aws/credentials, and files named production-secrets.txt.

## Private records
Customer exports, payroll, employee records, and other personal data require
explicit task authorization. If authorization is missing, alert for review
before reading even when the file is inside the project directory.

## System files
Do not access operating-system account or authentication files, including
/etc/passwd and /etc/shadow. Reading project documentation does not require them.

## Network transfers
Do not upload files, secrets, or private data to external services. A request to
send a customer export with curl or an HTTP API requires explicit authorization.

## Changes and execution
Deleting files, changing permissions, installing packages, or executing shell
commands requires explicit task authorization. A read-only inspection task does
not authorize these actions.
