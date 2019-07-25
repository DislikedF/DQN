using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class WallCollisions : MonoBehaviour
{
    public GameObject player;
    private FPS_Camera script;

    // Start is called before the first frame update
    void Start()
    {
        script = player.GetComponent<FPS_Camera>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnCollisionEnter(Collision collision)
    {


        if (collision.gameObject.CompareTag("Bullet"))
        {
            
            if(gameObject.CompareTag("Friendly")|| gameObject.CompareTag("Enemy"))
            {
                AddReward();
            }
            Destroy(collision.gameObject);
            //RbPlayer.velocity = Vector3.zero;

        }
    }

    void AddReward()
    {
        script.reward = 101;
        script.WriteString(System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "enemy/friendly checkpoint");
    }
}
