# LilLisa Server
## Description

Built using FastAPI, LilLisa Server is an AI question-and-answer program that enables users to ask natural language questions and receive insightful answers based on their personalized data.

LilLisa Server uses LanceDB as the vector database, providing blazingly fast retrieval times and LlamaIndex to handle complex queries.

LilLisa Server is able to grow its knowledge base and get better at answering user questions as more documentation is added and conversations are stored.

## Visuals

[Conversation within Slack](./visuals/conversation_slack.png)

[Conversation within Browser](./visuals/conversation_web.png)

## Installation

- Clone this project using this command:
  - git clone https://oauth2:&lt;YOUR_GITLAB_ACCESS_TOKEN&gt;@gitlab.com/radiant-logic-engineering/rl-datascience/lil-lisa.git
- Navigate to lil-lisa folder
- In the terminal, run "make condaenv". This will create a conda environment and install all necessary packages
- Select 'lillisa-server' as python interpreter

## Usage

<ins>IMPORTANT<ins>: If using VS Code, start main.py by using "Python Debugger: Debug using launch.json"
​
Integrate with Slack, FastHTML, or another application that handles user input. Run one of the above along with LilLisa_Server conccurently.

Slash commands are encrypoted and can only be used by admins specified in Slack.

Methods free to use are:

**/invoke**

Uses a session ID to retrieve past conversation history. Based on a query, it searches relevant documents in the knowledge base and retrieves multiple fewshot examples from the QA pairs database to help synthesis of a formatted answer. Queries are handled differently and depend on whether an expert is answering or not. ReACT agent handles the use of information given to intellignetly craft an answer.

**/record_endorsement**

Records an endorsemewnt, usually given when an answer is correct, by either a "user" or "expert". This is helpful when admins call the 'get_conversations' method and use it to create more golden QA pairs.

## Contributing

The project is not currently open for contributions.

### Requirements
- Docker container
- Python 3.11.9 
- RAM: 1.0 GB
- Size of Docker container: 11.2 GB

### Pushing to Cloud

For assistance with deploying to AWS Lambda, refer to this blog:
  - https://fanchenbao.medium.com/api-service-with-fastapi-aws-lambda-api-gateway-and-make-it-work-c20edcf77bff

## Support

Reach out to us if you have questions:
- Carlos Escobar (Slack: @Carlos Escobar, Email: cescobar@radiantlogic.com)
- Dhar Rawal (Slack: @Dhar Rawal, Email: drawal@radiantlogic.com)
- Unsh Rawal (Slack: @Unsh Rawal, Email: urawal@radiantlogic.com)

## Authors and acknowledgment

- Carlos Escobar
- Dhar Rawal
- Unsh Rawal
- Nico Guyot

## License

This project is currently closed source

## Project status

Under active development

## Socials
- [Link to Medium.com blog](https://medium.com/@carlos-a-escobar/deep-dive-into-the-best-chunking-indexing-method-for-rag-5921d29f138f)