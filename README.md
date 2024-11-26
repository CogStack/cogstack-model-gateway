# CogStack Model Gateway

The CogStack ModelGateway (CMG) is a service that provides a unified interface for accessing machine learning models deployed as standalone servers. It implements service discovery and enables
scheduling incoming tasks based on their priority, as well the state of the cluster. The project is designed to work with [Cogstack ModelServe](https://github.com/CogStack/CogStack-ModelServe) model
server instances and consists of two main components:

* **Model Gateway**: A RESTful API that provides a unified interface for accessing machine learning
  models deployed as standalone servers. The gateway is responsible for assigning a priority to each
  incoming task and publishing it to a queue for processing.
* **Task Scheduler**: A service that schedules queued tasks based on their priority and the state of
    the cluster. The scheduler is responsible for ensuring that tasks are processed in a timely
    manner and that the cluster is not overloaded.

CogStack ModelGateway comes with a persistence layer that stores information about scheduled tasks,
exposed through a REST API for visibility and monitoring.
