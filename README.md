# cricket-predictions

### Local usage
Requirements: anaconda
- bash deploy.sh
- bash test.sh

### Docker
Requirements: docker 
- docker build -t cricket_model docker/
- docker run -it cricket_model bash
- (inside container) python infer.py --test
