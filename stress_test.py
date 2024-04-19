from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def post_emotion(self):
        self.client.post("/", {
            "text": "I'm feeling very happy today!"
        })
