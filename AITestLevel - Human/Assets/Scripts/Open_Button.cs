using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class Open_Button : MonoBehaviour
{
    public Animator Anim;
    public GameObject Door;
    public GameObject player;


    private Collider DoorCollider;
    private FPS_Camera script;
    


    // Start is called before the first frame update
    void Start()
    {
        DoorCollider = Door.GetComponent<Collider>();
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
            Anim.SetTrigger("character_nearby");
            DoorCollider.enabled = false;
            AddReward();
            Destroy(collision.gameObject);
            //RbPlayer.velocity = Vector3.zero;

        }
    }

    void AddReward()
    {
        script.reward = 105;
        script.WriteString(System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "Door open checkpoint");
    }
}
