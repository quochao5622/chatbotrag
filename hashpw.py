import bcrypt 
  
# example password 
password = '123456'
  
# converting password to array of bytes 
bytes = password.encode('utf-8') 
  
# generating the salt 
salt = bcrypt.gensalt() 
  
# Hashing the password 
hash = bcrypt.hashpw(bytes, salt) 
  
print(hash)