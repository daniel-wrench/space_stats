def calc_dep(v2lu,primitives,derived,maxiter=1000):
   lgcl=[False]*len(v2lu)
   it=0
   while not all(lgcl):
      for i in v2lu:
         idx=v2lu.index(i)
         if i in primitives:
            lgcl[idx]=True
         elif i in derived:
            v2lu.pop(idx)
            if derived[i] not in v2lu:
               v2lu.append(derived[i])
               lgcl[idx]=True
            else:
               lgcl.pop(idx)
      it+=1
      if it > maxiter: 
         print('Could not resolve the dependencies\
                in {0} iterations'.format(maxiter))
         break
   return v2lu

