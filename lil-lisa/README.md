# Lil Lisa
## Description

Lil Lisa is an application that is responsible for integrating with Slack and handling incoming user events. It acts as a bridge between Slack's platform and the conversational AI experience provided by API calls to a concurrently-running FastAPI application.  

Lil Lisa provides several slash commands for admin use to help improve the knowledge base.

## Visuals

![Simple conversation](./visuals/simple_conversation.png)
![Architecture/Diagram](./visuals/diagram.png)

## Installation

Simply add Lil Lisa to your Slack workspace

1. Open Slack
2. In the bottom left, under Apps, click 'Add apps'
3. Search for "rocknbot" and select it.

[Click here if you need further instruction on adding an app to Slack](https://slack.com/help/articles/202035138-Add-apps-to-your-Slack-workspace)

Setup Lil Lisa Server and have both applications running at the same time.
​
## Usage

For use in a channel, simply call **@rocknbot** in the channel, and it will return with *'Processing...'* and subsequently the answer to the question.

For IDDM queries, refer to the "lil-elvis" channel.
For IDA queries, refer to the "lil-lisa" channel.

The slash commands available for admin use include:

**/get_golden_qa_pairs**

Retrieves QA pairs stored in the appropriate github repository

**/update_golden_qa_pairs**

Ingests changes made to the QA pairs github repository by deleting the previous LanceDB tables and recreating with up-to-date data

**/get_conversations [endorsed_by]**

Retrieves conversations from a local folder that were endorsed by [endorsed_by]

**/rebuild_docs**

Ingests changes made to the knowledge base (github repository) by deleting the previous LanceDB tables and recreating with up-to-date data

## Contributing

The project is not currently open for contributions.

### Requirements

- Docker container
- Python 3.11.9
- RAM: 0.4 GB
- Size of Docker container: 1.4 GB

### Setup dev environment

- Clone this project using this command:
  - git clone https://oauth2:&lt;YOUR_GITLAB_ACCESS_TOKEN&gt;@gitlab.com/radiant-logic-engineering/rl-datascience/lil-lisa.git
- Navigate to lil-lisa folder
- In the terminal, run "make _build-local". This will create a conda environment and install all necessary packages
- Select 'lil-lisa' as python interpreter

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

## Future Enhancements

- Version-specific answers
- Response streaming
- Return screenshots/images with answers
- Allow users to provide screenshots with their questions
- Let users Direct Message the bot for Question-Answering
- Refine the handling of expert answers in answer synthesis