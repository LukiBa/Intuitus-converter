# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:27:14 2020

@author: lukas
"""
import numpy as np
from typing import Tuple 
import copy



class floatX():
    def __init__(self,_sign,_exp,_mantissa,exp_width=4,mantissa_width=4, sign_type=np.int8, exp_type=np.int8, mantissa_type=np.uint8): 
        self._mantissa_type = mantissa_type
        self._exp_type = exp_type
        self._sign_type = sign_type   
        
        self._mantissa_width = mantissa_width
        self._exp_width = exp_width
        
        self._sign = _sign
        self._exp = _exp
        self._mantissa = _mantissa

    @property
    def sign(self):
        return self._sign
    @sign.setter
    def sign(self,val):
        if isinstance(self._exp,np.ndarray):
            self._sign = val.astype(self._sign_type).copy()
        else:
            self._sign = val

    @property
    def exp(self):
        return self._exp
    @exp.setter
    def exp(self,val):
        if isinstance(self._exp,np.ndarray):
            self._exp = val.astype(self._exp_type).copy()
        else:
            self._exp = val
            
    @property
    def mantissa(self):
        return self._mantissa
    @mantissa.setter
    def mantissa(self,val):
        if isinstance(self._exp,np.ndarray):
            self._mantissa = val.astype(self._mantissa_type).copy()    
        else:
            self._mantissa = val
    
    @property 
    def signed_mantissa(self):
        return np.int8(np.where(self._sign==0,self._mantissa,(-1)*self._mantissa))
    
    @property 
    def shape(self):
        if isinstance(self._exp,np.ndarray):
            return self._exp.shape
        else:
            return 0
    @property
    def numpy(self):
        return self.to_float32
    @numpy.setter
    def numpy(self,value):
        self.set(value)
    def __getitem__(self, item):
        return floatX(self.sign[item],self.exp[item],self.mantissa[item],self._exp_width,self._mantissa_width) 
    def __setitem__(self, item, value):
            self.sign[item] = value.sign
            self.exp[item] = value.exp
            self.mantissa[item] = value.mantissa    
    def get_sign(self):
        return self.exp
    def get_exp(self):
        return self.exp
    def get_mantissa(self):
        return self._mantissa    

    def get_element(self,element : Tuple):
        return floatX(self.sign[element],self.exp[element],self._mantissa[element],self._exp_width,self._mantissa_width)                  
    def to_float32(self):
        return np.float32(np.where(self.sign==0,self._mantissa*2.0**((-1)*self.exp-self._mantissa_width),(-1)*self._mantissa*2.0**((-1)*self.exp-self._mantissa_width)))        
    def to_binary(self): 
        return np.bitwise_or(np.int32(self._mantissa),
                             np.bitwise_or(np.left_shift(
                                             np.bitwise_and(np.int32(2.0**self._exp_width-1),np.int32(self.exp)),self._mantissa_width),
                                             np.left_shift(np.int32(self.sign),self._mantissa_width+self._exp_width)))
    
    def set(self,data):
        value = np.abs(data)
        exp = np.int8(np.where(value==0.0,(2.0**self._exp_width-1)-1,(-1)*np.floor(np.log2(value))))
        exp = np.int8(np.clip(exp,(-1)*(2.0**(self._exp_width-1)),2.0**(self._exp_width-1)-1))                
        _mantissa = np.uint8(np.around(value*2.0**(np.int32(np.abs(exp))+self._mantissa_width)))
        _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**self._mantissa_width)-1)) 
        self.sign = np.int8(np.where(data<0.0,1,0))
        self.exp = exp
        self._mantissa = _mantissa
        self.shape = exp.shape
    def get_quantization(self):
        return self._exp_width,self._mantissa_width
    def __add__(self,data):
        if not isinstance(data,floatX):
            data = floatX(data,self._mantissa_width,self._exp_width)  
  
        exp_diff = self.exp - data.exp
        exp = np.where(exp_diff<=0,self.exp,data.exp)
        self_mantissa = np.where(self.sign==0,np.int8(self._mantissa),(-1)*np.int8(self._mantissa))
        data_mantissa = np.where(self.sign==0,np.int8(data.mantissa),(-1)*np.int8(data.mantissa))
        _mantissa = np.where(exp_diff<=0,np.right_shift(data_mantissa,exp_diff)+self_mantissa,np.right_shift(self_mantissa,exp_diff)+data_mantissa)
        sign = np.int8(np.where(_mantissa<0,1,0))
        _mantissa = np.uint8(np.abs(_mantissa))
        exp = np.where(_mantissa>=2.0**self._mantissa_width,exp-1,exp)
        if (np.abs(exp) >= 2.0**(self._exp_width-1)).any():
            raise OverflowError
        _mantissa = np.where(_mantissa>=2.0**self._mantissa_width,np.right_shift(self._mantissa,1),_mantissa)
        return floatX(sign,exp,_mantissa,self._mantissa_width,self._exp_width)                     
    def __sub__(self,data):
        if not isinstance(data,floatX):
            data = floatX(data,self._mantissa_width,self._exp_width)  
  
        exp_diff = self.exp - data.exp
        exp = np.where(exp_diff<=0,self.exp,data.exp)
        self_mantissa = np.where(self.sign==0,np.int8(self._mantissa),(-1)*np.int8(self._mantissa))
        data_mantissa = np.where(self.sign==0,np.int8(data.mantissa),(-1)*np.int8(data.mantissa))
        _mantissa = np.where(exp_diff<=0,np.right_shift(data_mantissa,exp_diff)-self_mantissa,np.right_shift(self_mantissa,exp_diff)-data_mantissa)
        sign = np.int8(np.where(_mantissa<0,1,0))
        _mantissa = np.uint8(np.abs(_mantissa))
        exp = np.where(_mantissa>=2.0**self._mantissa_width,exp-1,exp)
        if (np.abs(exp) >= 2.0**(self._exp_width-1)).any():
            raise OverflowError
        _mantissa = np.where(_mantissa>=2.0**self._mantissa_width,np.right_shift(self._mantissa,1),_mantissa)
        return floatX(sign,exp,_mantissa,self._mantissa_width,self._exp_width)  
    
    def __mul__(self,data):
        if not isinstance(data,floatX):
            data = floatX(data,self._mantissa_width,self._exp_width)  
            
        exp, _mantissa = self._mul_unsigned(data.exp,data._mantissa)
        sign = np.bitwise_xor(self.sign,data.sign)
        return floatX(sign,exp, _mantissa,self._mantissa_width,self._exp_width)
    def __rmul__(self,data):
        return self.__mul__(data)
    def _mul_unsigned(self,exp_data,mantissa_data):
        exp = self.exp + exp_data
        exp = np.where(exp>=2.0**(self._exp_width-1),2.0**(self._exp_width-1)-1,exp)
        _mantissa = np.where(exp>=2.0**(self._exp_width-1),0,self._mantissa*mantissa_data)
        quant = np.uint8(np.ceil(np.log2(_mantissa))-self._mantissa_width) 
        _mantissa = np.where(quant>0,np.right_shift(_mantissa,quant),_mantissa)
        exp = np.where(quant>0,exp-quant,_mantissa)
        if (exp <= (-1)*2.0**(self._exp_width-1)).any():
            OverflowError("Overflow in multiplication")
        return exp, _mantissa  
    def max_pool_2d(self,channel_last = False):
        if channel_last == False:
            a = self[...,::2]
            b = self[...,1::2]
            res = self._pool(a, b)
            a = res[...,::2,:]
            b = res[...,1::2,:]
            res = res._pool(a, b)
            return res
        else:
            a = self[...,::2,:]
            b = self[...,1::2,:]
            res = self._pool(a, b)
            a = res[...,::2,:,:]
            b = res[...,1::2,:,:]
            res = res._pool(a, b)
            return res           
        
    def _pool(self,a,b):
        res = copy.deepcopy(b)
        cond = a.exp < b.exp
        res[cond] = a[cond]
        cond = np.logical_and(a.exp == b.exp, a.mantissa > b.mantissa)
        res[cond] = a[cond]
        return res

    def normalize(self):
        unshifted_mantissa = self._mantissa.copy()
        mantissa_shift = np.where(self._mantissa>=0,np.int8(np.ceil(np.log2(self._mantissa+1)))-self._mantissa_width,np.int8(np.ceil(np.log2(np.abs(self._mantissa))))-self._mantissa_width)
        mantissa_shift = np.where(self.exp==2.0**(self._exp_width)-1,np.clip(mantissa_shift,0,1),mantissa_shift)
        mantissa_shift = np.where(self.exp==2.0**(self._exp_width)-2,np.clip(mantissa_shift,-1,1),mantissa_shift)
        mantissa_shift = np.clip(mantissa_shift,-2,1)
        self._mantissa = np.where(mantissa_shift>=0,np.right_shift(self._mantissa,mantissa_shift),np.left_shift(self._mantissa,np.abs(mantissa_shift)))
        self.exp = np.int8(self.exp - mantissa_shift)
        if (self._mantissa>=2.0**(self._mantissa_width)).any():
            raise OverflowError("Mantissa overflow in MACC: {}{}".format(unshifted_mantissa,mantissa_shift)) 

class float8(floatX):
    def __init__(self,*data):
        mantissa_width = 4
        exp_width = 3        
        if len(data) == 1:  
            if isinstance(data[0],floatX):
                sign = data[0].sign
                exp= data[0].exp
                _mantissa = data[0].mantissa
            else:
                sign = np.int8(np.where(data[0]<0.0,1,0))
                value = np.abs(data[0])
                exp = np.where(value==0.0,(2.0**exp_width)-1,np.floor((-1)*np.log2(value)))
                exp = np.int8(np.clip(exp,0,2.0**(exp_width)-1))                
                _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+mantissa_width)))
                _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**mantissa_width)-1)) 
            super(float8,self).__init__(sign,exp,_mantissa,exp_width,mantissa_width)
        
        elif len(data) == 2:
            sign = np.zeros(data[0].shape,dtype=np.uint8)
            super(float8,self).__init__(sign,data[0],data[1],exp_width,mantissa_width)  
        elif len(data) == 3:
            super(float8,self).__init__(data[0],data[1],data[2],exp_width,mantissa_width)   
        else:
            raise AttributeError("Wrong number of arguments")  
    def set(self,val):
        sign = np.int8(np.where(val<0.0,1,0))
        value = np.abs(val)
        exp = np.where(value==0.0,(2.0**self._exp_width)-1,np.floor((-1)*np.log2(value)))
        exp = np.int8(np.clip(exp,0,2.0**(self._exp_width)-1))                
        _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+self._mantissa_width)))
        _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**self._mantissa_width)-1)) 
        self.sign = sign
        self.exp = exp
        self._mantissa = _mantissa
        
    def __setitem__(self, item, value):
            self.sign[item] = value.sign
            self.exp[item] = value.exp
            self._mantissa[item] = value.mantissa
    def __getitem__(self, item):
        return float8(self.sign[item],self.exp[item],self._mantissa[item]) 
    

class float9(floatX):
    def __init__(self,*data):
        mantissa_width = 4
        exp_width = 4        
        if len(data) == 1:  
            if isinstance(data[0],floatX):
                sign = data[0].sign
                exp= data[0].exp
                _mantissa = data[0].mantissa
            else:
                value = np.abs(data[0])
                sign = np.int8(np.where(data[0]<0.0,1,0))
                exp = np.int8(np.where(value==0.0,(2.0**(exp_width-1))-1,np.floor((-1)*np.log2(value))))
                exp = np.int8(np.clip(exp,(-1)*(2.0**(exp_width-1)),2.0**(exp_width-1)-1))                
                _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+mantissa_width)))
                _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**mantissa_width)-1)) 
            super(float9,self).__init__(sign,exp,_mantissa,exp_width,mantissa_width)
            
        elif len(data) == 3:
            super(float9,self).__init__(data[0],data[1],data[2],exp_width,mantissa_width)   
        else:
            raise AttributeError("Wrong number of arguments")

class float12(floatX):
    def __init__(self,*data):
        mantissa_width = 7
        exp_width = 4        
        if len(data) == 1:  
            if isinstance(data[0],floatX):
                sign = data[0].sign
                exp= data[0].exp
                _mantissa = data[0].mantissa
            else:
                value = np.abs(data[0])
                sign = np.int8(np.where(data[0]<0.0,1,0))
                exp = np.where(value==0.0,(2.0**exp_width)-1,np.floor((-1)*np.log2(value)))
                exp = np.int8(np.clip(exp,(-1)*(2.0**(exp_width-1)),2.0**(exp_width)-1))                
                _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+mantissa_width)))
                _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**mantissa_width)-1)) 
            super(float12,self).__init__(sign,exp,_mantissa,exp_width,mantissa_width)
            
        elif len(data) == 3:
            super(float12,self).__init__(data[0],data[1],data[2],exp_width,mantissa_width)   
        else:
            raise AttributeError("Wrong number of arguments")

class float6_b(floatX):
    def __init__(self,*data):
        mantissa_width = 3
        exp_width = 2        
        if len(data) == 1:  
            if isinstance(data[0],floatX):
                sign = data[0].sign
                exp= data[0].exp
                _mantissa = data[0].mantissa
            else:
                sign = np.int8(np.where(data[0]<0.0,1,0))
                value = np.abs(data[0])
                exp = np.where(value==0.0,(2.0**exp_width)-1,np.floor((-1)*np.log2(value)))
                exp = np.int8(np.clip(exp,0,2.0**(exp_width)-1))                
                _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+mantissa_width)))
                _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**mantissa_width)-1)) 
            super(float6_b,self).__init__(sign,exp,_mantissa,exp_width,mantissa_width)
            
        elif len(data) == 3:
            super(float6_b,self).__init__(data[0],data[1],data[2],exp_width,mantissa_width)   
        else:
            raise AttributeError("Wrong number of arguments")
    def __getitem__(self, item):
        return float6_b(self.sign[item],self.exp[item],self._mantissa[item])
    def set(self,data):
        value = np.abs(data)
        exp = np.int8(np.where(value==0.0,(2.0**self._exp_width)-1,np.floor((-1)*np.log2(value))))
        exp = np.int8(np.clip(exp,0,(2.0**self._exp_width)-1))                
        _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+self._mantissa_width)))
        _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**self._mantissa_width)-1)) 
        self.sign = np.int8(np.where(data<0.0,1,0))
        self.exp = exp
        self._mantissa = _mantissa
        self.shape = exp.shape
    def get_element(self,element : Tuple):
        return float6_b(self.sign[element],self.exp[element],self._mantissa[element])  

class float6(floatX):
    def __init__(self,*data):
        mantissa_width = 4
        exp_width = 1        
        if len(data) == 1:  
            if isinstance(data[0],floatX):
                sign = data[0].sign
                exp= data[0].exp
                _mantissa = data[0].mantissa
            else:
                sign = np.int8(np.where(data[0]<0.0,1,0))
                value = np.abs(data[0])
                exp = np.where(value==0.0,(2.0**exp_width)-1,np.floor((-1)*np.log2(value)))
                exp = np.int8(np.clip(exp,0,2.0**(exp_width)-1))                
                _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+mantissa_width)))
                _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**mantissa_width)-1)) 
            super(float6,self).__init__(sign,exp,_mantissa,exp_width,mantissa_width)
            
        elif len(data) == 3:
            super(float6,self).__init__(data[0],data[1],data[2],exp_width,mantissa_width)   
        else:
            raise AttributeError("Wrong number of arguments")
    def __getitem__(self, item):
        return float6(self.sign[item],self.exp[item],self._mantissa[item])
    def __setitem__(self, item, value):
            self.sign[item] = value.sign
            self.exp[item] = value.exp
            self._mantissa[item] = value.mantissa
            
    def set(self,data):
        value = np.abs(data)
        exp = np.int8(np.where(value==0.0,(2.0**self._exp_width)-1,np.floor((-1)*np.log2(value))))
        exp = np.int8(np.clip(exp,0,(2.0**self._exp_width)-1))                
        _mantissa = np.uint8(np.around(value*2.0**(np.int32(exp)+self._mantissa_width)))
        _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**self._mantissa_width)-1)) 
        self.sign = np.where(data<0.0,1,0)
        self.exp = exp
        self._mantissa = _mantissa
        self.shape = exp.shape
    def get_element(self,element : Tuple):
        return float6(self.sign[element],self.exp[element],self._mantissa[element])  

class fixed4(floatX):
    def __init__(self,val,fixed_exp : int = 3):
        mantissa_width = 3
        exp_width = 0   
        
        value = np.abs(val)
        _mantissa = np.round(2.0**(fixed_exp+mantissa_width)*value)
        _mantissa = np.uint8(np.clip(_mantissa,0,2.0**(mantissa_width)-1))
        sign = np.where(val<0.0,1,0).astype(np.int8)
        
        super(fixed4,self).__init__(sign,fixed_exp,_mantissa,exp_width,mantissa_width)
    
    def set(self,val):
        value = np.abs(val)
        _mantissa = np.round(2.0**(self.exp+self._mantissa_width-1)*value)
        _mantissa = np.uint8(np.clip(_mantissa,0,2.0**(self._mantissa_width-1)-1))
        self.sign = np.where(val<0.0,1,0).astype(np.int8)        
        self._mantissa = _mantissa
    
    def __getitem__(self, item):
        return fixed4(self._mantissa[item])
    def get_element(self,element : Tuple):
        return fixed4(self.sign[element],self._mantissa[element])   
    def to_binary(self): 
        return np.where(self.sign==0,np.int32(self._mantissa),np.int32(self._mantissa)*(-1))
    # def to_float32(self):
    #     return np.float32(self.mantissa*2.0**((-1.0)*self.exp-self._mantissa_width))         
    
class ufloat8(floatX):
    def __init__(self,*data):
        mantissa_width = 4
        exp_width = 4
        if len(data) == 1:  
            if isinstance(data[0],floatX):
                sign = 0
                exp = np.where(sign == 0,data[0].exp,2.0**(exp_width-1)-1)
                _mantissa = np.where(sign == 0,data[0]._mantissa,0)
            else:
                sign = np.uint8(np.where(data[0]<0.0,1,0))
                value = np.abs(data[0])
                exp = np.where(value==0.0,(2.0**exp_width)-1,np.floor((-1)*np.log2(value)))
                exp = np.where(sign == 0,exp,2.0**(exp_width-1)-1)
                exp = np.int8(np.clip(exp,(-1)*(2.0**(exp_width-1)),2.0**(exp_width-1)-1))                
                _mantissa = np.uint8(np.where(sign == 0,np.around(value*2.0**(np.int32(exp)+mantissa_width)),0))
                _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**mantissa_width)-1)) 
                sign = np.zeros(exp.shape,dtype=np.uint8)
            super(ufloat8,self).__init__(sign,exp,_mantissa,exp_width,mantissa_width)
            
        elif len(data) == 2:
            super(ufloat8,self).__init__(0,data[0],data[1],exp_width,mantissa_width)   
        else:
            raise AttributeError("Wrong number of arguments")    
    def __getitem__(self, item):
         return ufloat8(self.exp[item],self._mantissa[item])
    def get_exp(self):
        return self.exp
    def get_mantissa(self):
        return self._mantissa                
    def to_float32(self):
        return np.float32((self._mantissa)*2.0**((-1)*self.exp-self._mantissa_width))        
    def get_element(self,element : Tuple):
        return ufloat8(self.sign[element],self.exp[element],self._mantissa[element])  
    def set(self,data):
        sign = np.uint8(np.where(data<0.0,1,0))
        value = np.abs(data[0])
        exp = np.int8(np.where(data==0.0,2.0**(self._exp_width-1)-1,np.floor((-1)*np.log2(value))))
        exp = np.where(sign == 0,exp,2.0**(self._exp_width-1)-1)
        exp = np.int8(np.clip(exp,(-1)*(2.0**(self._exp_width-1)),2.0**(self._exp_width-1)-1))                
        _mantissa = np.uint8(np.where(sign == 0,np.around(data*2.0**(np.int32(exp)+self._mantissa_width)),0))
        _mantissa = np.uint8(np.clip(_mantissa,0,(2.0**self._mantissa_width)-1)) 
        self.sign = np.zeros(exp.shape,dtype=np.uint8)
        self.exp = exp
        self._mantissa = _mantissa
        self.shape = exp.shape          
                        

class MACC():
    def __init__(self,bias):
        self._accum_width = 8
        self._exp_width = 4
        bias_exp = 3
        bias_width = 4
        bias_sign = np.where(bias<0, 1 , 0)
        bias = np.abs(bias)
        mantissa = np.round(2.0**(bias_exp+bias_width-1)*bias)
        mantissa = np.uint8(np.clip(mantissa,0,2.0**(bias_width-1)-1))  
        mantissa = np.left_shift(np.int32(mantissa),self._accum_width-bias_width+1)
        
        self.sign = bias_sign
        self.exp = bias_exp
        
        self.mantissa = np.where(bias_sign==0, mantissa, (-1) * mantissa)
        self.shape = self.mantissa.shape

    def __call__(self,A,B): 
        if not (isinstance(A,floatX) and isinstance(A,floatX)):
            raise AttributeError("macc requires arguments of type flaotX")
        
        exp = np.int32(A.exp) + np.int32(B.exp)
        exp_diff = exp - self.exp
        a_mantissa = np.where(exp_diff >=0, np.right_shift(A.mantissa,exp_diff),A.mantissa)
        p_mantissa = np.where(exp_diff >=0, self.mantissa,np.right_shift(self.mantissa,np.abs(exp_diff)))
        
        m_mantissa = np.int32(a_mantissa) * np.int32(B.mantissa)
        sign = np.bitwise_xor(A.sign,B.sign)
        
        self.exp = np.where(exp_diff >=0,self.exp,exp)
        self.mantissa = np.where(sign==0,p_mantissa + m_mantissa, p_mantissa - m_mantissa)
        
        #self.exp = np.int8(np.where(self.mantissa>=2**self._accum_width,self.exp-1,self.exp))  
        #self.mantissa = np.uint8(np.where(self.mantissa>=2**self._accum_width,np.right_shift(self.mantissa,1),self.mantissa)) 
        unshifted_mantissa = self.mantissa
        mantissa_shift = np.where(self.mantissa>=0,np.int32(np.ceil(np.log2(self.mantissa+1)))-self._accum_width,np.int32(np.ceil(np.log2(np.abs(self.mantissa))))-self._accum_width)
        mantissa_shift = np.where(self.exp==2**(self._exp_width-1)-1,np.clip(mantissa_shift,0,1),mantissa_shift)
        mantissa_shift = np.where(self.exp==2**(self._exp_width-1)-2,np.clip(mantissa_shift,-1,1),mantissa_shift)
        mantissa_shift = np.clip(mantissa_shift,-2,1)
        self.mantissa = np.where(mantissa_shift>=0,np.right_shift(self.mantissa,mantissa_shift),np.left_shift(self.mantissa,np.abs(mantissa_shift)))
        self.exp -= mantissa_shift
        if (self.mantissa>=2**(self._accum_width)).any():
            raise OverflowError("Mantissa overflow in MACC: {}{}".format(unshifted_mantissa,mantissa_shift))          
        
        self.exp = np.int8(np.where(self.exp>=2**(self._exp_width-1),2**(self._exp_width-1)-1,self.exp)) 
        self.mantissa = np.where(self.exp>=2**(self._exp_width-1),0,self.mantissa) 
        if (self.exp<=(-1)*2**(self._exp_width-1)).any():
            raise OverflowError("Exponent overflow in MACC: Exponent: {}".format(self.exp))
            # self.exp = np.int8(np.where(self.exp<=(-1)*2**(self._exp_width-1),(-1)*2**(self._exp_width-1)-1,self.exp)) 
            # self.mantissa = np.uint8(np.where(self.exp<=(-1)*2**(self._exp_width-1),2**(self._accum_width)-1,self.mantissa)) 
            # print("Exponent overflow in MACC: Exponent: {}".format(self.exp))
        
        self.shape = self.mantissa.shape
        
    def __getitem__(self, item):
        sign = np.where(self.mantissa<0,1,0)
        mantissa = np.abs(self.mantissa)
        return floatX(sign[item],self.exp[item],mantissa[item],self._exp_width,self._accum_width)
    def normalize(self):
        unshifted_mantissa = self.mantissa
        mantissa_shift = np.where(self.mantissa>=0,np.int8(np.ceil(np.log2(self.mantissa+1)))-self._accum_width,np.int8(np.ceil(np.log2(np.abs(self.mantissa))))-self._accum_width)
        mantissa_shift = np.where(self.exp==2**(self._exp_width-1)-1,np.clip(mantissa_shift,0,1),mantissa_shift)
        mantissa_shift = np.where(self.exp==2**(self._exp_width-1)-2,np.clip(mantissa_shift,-1,1),mantissa_shift)
        mantissa_shift = np.clip(mantissa_shift,-2,1)
        self.mantissa = np.where(mantissa_shift>=0,np.right_shift(self.mantissa,mantissa_shift),np.left_shift(self.mantissa,np.abs(mantissa_shift)))
        self.exp = np.uint8(self.exp - mantissa_shift)
        if (self.mantissa>=2**(self._accum_width)).any():
            raise OverflowError("Mantissa overflow in MACC: {}{}".format(unshifted_mantissa,mantissa_shift))   
            
    def to_float32(self):
        return np.float32(self.mantissa*2**((-1.0)*self.exp-self._accum_width))
    def get_quantization(self):
        return self._exp_width,self._mantissa_width
    def get_floatX(self):
        sign = np.where(self.mantissa<0,1,0)
        mantissa = np.abs(self.mantissa)
        return floatX(sign,self.exp,mantissa,self._exp_width,self._accum_width) 
    def to_float8(self):
        sign = np.where(self.mantissa<0,1,0)
        mantissa = np.abs(self.mantissa)
        mantissa = np.right_shift(mantissa,self._accum_width-4)      
        return float8(sign,self.exp,mantissa) 
    def set_value(self,sign,exp,mantissa):
        self.sign = sign
        self.mantissa = mantissa
        self.exp = exp
    
class MACC_slim(MACC):
    def __init__(self,bias):
        self._accum_width = 8
        self._exp_width = 4 # in real exp_width = 3 --> unsigned        
        bias_exp = 3
        bias_width = 4
        bias_sign = np.where(bias<0, 1 , 0)
        bias = np.abs(bias)
        bias_mantissa = np.int8(np.around(2**(bias_exp+bias_width-1)*bias))
        bias_mantissa = np.clip(bias_mantissa,(-1)*2**(bias_width-1),2**(bias_width-1)-1)
        
        #bias_mantissa = np.where(bias>(2**(bias_width-1)-1)*2**(-bias_width-bias_exp+1),2**(bias_width-1)-1,
        #                         np.int8(np.around(2**(bias_exp+bias_width-1)*np.abs(bias))))
        #bias_mantissa = np.clip(bias_mantissa,0,2**(bias_width-1)-1)
        self.sign = bias_sign
        self.exp = bias_exp
        self.mantissa = np.where(bias_sign==0,np.left_shift(bias_mantissa,self._accum_width-bias_width+1),
                                             (-1) * np.left_shift(bias_mantissa,self._accum_width-bias_width+1))

        self.shape = self.mantissa.shape
        self.feedback_width = 5
        self.mask = np.int8(np.left_shift(2**self.feedback_width-1,self._accum_width-self.feedback_width+1))
        self.mantissa = np.where(self.mantissa < 0,np.bitwise_or(self.mantissa,np.bitwise_not(self.mask)),np.bitwise_and(self.mantissa,self.mask))
        self.mantissa = np.where(self.mantissa < 0,np.bitwise_and(self.mantissa,np.bitwise_not(3)),np.bitwise_or(self.mantissa,3))

    def __call__(self,A,B):       
        super(MACC_slim,self).__call__(A,B)
        self.mantissa = np.bitwise_and(self.mantissa,self.mask)
        self.mantissa = np.where(self.mantissa < 0,np.bitwise_or(self.mantissa,np.bitwise_not(self.mask)),np.bitwise_and(self.mantissa,self.mask))
        self.mantissa = np.where(self.mantissa < 0,np.bitwise_and(self.mantissa,np.bitwise_not(3)),np.bitwise_or(self.mantissa,3))
        self.exp = np.uint8(self.exp)