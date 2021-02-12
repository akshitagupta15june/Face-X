/* Copyright 2016 Michael Sladoje and Mike Schälchli. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package ch.zhaw.facerecognitionlibrary.Helpers;

import java.util.HashMap;
import java.util.Map;

/***************************************************************************************
 *    Title: One-to-one mapping data structure (A,B) with getKey(B) in O(1)?
 *    Author: japreiss
 *    Date: 22.06.2012
 *    Code version: -
 *    Availability: http://stackoverflow.com
 *
 ***************************************************************************************/

public class OneToOneMap<Key, Value> {
    private Map<Key, Value> keyToVal;
    private Map<Value, Key> valToKey;

    public OneToOneMap() {
        this.keyToVal = new HashMap<>();
        this.valToKey = new HashMap<>();
    }

    public void put(Key k, Value v) {
        if (!keyToVal.containsKey(k) && !valToKey.containsKey(v)) {
            keyToVal.put(k, v);
            valToKey.put(v, k);
        }
    }

    public boolean containsKey(Key k){
        return keyToVal.containsKey(k);
    }

    public Value getValue(Key k){
        return keyToVal.get(k);
    }

    public Key getKey(Value v){
        return valToKey.get(v);
    }

    public int size(){
        return keyToVal.size();
    }

    public Map getKeyToValMap(){
        return keyToVal;
    }
}
