# Quickstart Manual for RCAccelerator Users

There are two ways to use the RCAccelerator chatbot:

1. Connecting to our OpenShift cluster
2. Running a container on your local machine

## Method 1: The UI app on our OpenShift cluster

### 1. Run credentials extraction sript
Run the credentials extraction sript from the CLI (VPN connection is required)

   ```bash
   curl -O https://url.corp.redhat.com/update-the-link-when-pr-is-merged-py
   chmod +x update-the-link-when-pr-is-merged-py
   ./update-the-link-when-pr-is-merged-py
   ```

Your username and password will be printed by the script, allowing you to access the chatbot UI.

### 2. Navigate to the chatbot page

Use the credentials that were obtained in the previous step to access [https://url.corp.redhat.com/ui-rcaccelerator](https://url.corp.redhat.com/ui-rcaccelerator). You will see the RCAccelerator chat interface after logging in.

## Method 2: A local instance of a container
You can run chat application localy that will connect to backend services running on our OpenShift cluster.
VPN connection is required here as well.

### 0. Prerequisites
- OpenShift CLI (`oc`) installed
- Podman installed
- Proper permissions to update `/etc/hosts` (may require sudo)



### 1. Clone the Repository

```bash
git clone https://github.com/RCAccelerator/chatbot.git
cd chatbot
```

### 2. Create an Environment File

Use the extraction script to generate your `.env` file from the OpenShift manifests:
Get [extract_env.py](https://url.corp.redhat.com/env-generator) and run:

```bash
python3 extract_env.py [app.yaml](https://url.corp.redhat.com/app-yaml)
```

### 2. Update Hosts File

Get the endpoints of the backend services running on our OpenShift cluster

```bash
export KUBECONFIG=<path-to-repo>/openshift-manifests/auth/kubeconfig
oc get Services -n rcaccelerator
```

and update `/etc/hosts` file with the IPs obtained:

```bash
# Example (replace with IPs from the output obtained earlie)
X.Y.Z.66  rcaccelerator-postgresql
X.Y.Z.102 rcaccelerator-vectordb
```

### 3. Build the Container Image

```bash
podman build -t chatbot:latest -f Containerfile .
```

### 4. Run the Application

Start the chatbot application on your local machine:

```bash
podman run --rm --env-file .env -it -p 8000:8000 chatbot:latest
```

and navigate to:

```
http://localhost:8000
```

## Using RCAccelerator Effectively

### Best Practices

1. **Be Specific**: Give specific details regarding the CI failure you're looking into.

   Good: "the fault 'XYZ' caused Zuul job X to fail. Please include a list of Jira tickets that may be connected to the problem.

   The phrase "My build is broken" is less useful.

2. **Include Context**: Enhance your question/query with error messages, logs and any other relevant information

3. **Ask Follow-Up Questions**: Ask clarifying questions if the first response doesn't address your problem.


### Troubleshooting

If you experience problems with RCAccelerator:

1. **Response Quality Issues**: Consider making your query more particular. Divide complicated questions into smaller, more targeted ones
   Find relevant Jira ticket to make sure RCAccelerator inventory contains details about your question.

2. **Container Startup Issues**: Obtain container logs for error messages.

For additional support, contact the RCAccelerator team.
