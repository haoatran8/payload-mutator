from xss_detector.check import check_xss

def main():
    
    payload = "<script> alert(1); </script>"  
    result = check_xss(payload)
    print(result)  

if __name__ == "__main__":
    main()